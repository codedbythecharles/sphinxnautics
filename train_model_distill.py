import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
#from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.data import DataLoader
from functools import partial
from datasets import load_from_disk,load_dataset
#from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import helpers
import argparse
import re
import ast
from collections import defaultdict
import json, os
assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
from hf_evaluation import evaluate_model_on_dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple
#torch.set_float32_matmul_precision('high')

#model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
total_batch_size = 2**17 # for a 7B model target 2M per GPT3 paper but we are fine-tuning so it can be less. 
B = 1 # micro batch size
T = 16384 # average sequence length
ddp_world_size=1
gradient_accumulation_steps = total_batch_size // (B * T * ddp_world_size)


"""
#First Create success ids from json files and save locally
cd /root/workspace/jsons_train_32B
python3 extract_success_ids.py --dir .


# Convert a dataset (local)
python3 filter_success_ds.py \
  --csv success_best_overall.csv \
  --dataset_in /path/to/dataset_dir \
  --dataset_out filtered_success_ds

#Dataset from the Hub:

python3 filter_success_ds.py \
  --csv success_best_overall.csv \
  --dataset_in your_org/your_dataset \
  --split train \
  --dataset_out filtered_success_ds

#Then 
from datasets import load_from_disk
ds = load_from_disk("filtered_success_ds")
print(ds[0]["problem_id"], ds[0]["trials_to_success"])

#now build a has
python compare_ds.py build \
  --ref_ds filtered_success_ds \
  --out_index problem_index_minilm \
  --ref_out filtered_success_ds_hashed

#and compare a new dataset against the hashed one  
python comparse_ds.py compare \
  --index_dir problem_index_minilm \
  --query_ds new_dataset_dir \
  --out_ds new_dataset_annotated \
  --threshold 0.92 \
  --top_k 10

"""
writer = SummaryWriter(log_dir="runs/exp12")
checkpoint_dir = "checkpoints4"
os.makedirs(checkpoint_dir, exist_ok=True)


# ---------- helpers ----------------------------------------------------------
def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def shard_dataset(ds, world_size, rank):
    # contiguous=True keeps ranges, good for load‑latency
    return ds.shard(num_shards=world_size, index=rank, contiguous=True)

def gather_results(rank_results):
    world_size = dist.get_world_size()
    gathered   = [None] * world_size
    dist.all_gather_object(gathered, rank_results)
    # flatten
    if dist.get_rank() == 0:
        merged = []
        for part in gathered:
            merged.extend(part)
        return merged
    return None
# -----------------------------------------------------------------------------

def distributed_eval(model, val_ds, tokenizer, cfg):
    rank, world_size = 0,1#dist.get_rank(), dist.get_world_size()

    # 1. dataset slice for *this* rank
    val_ds_rank = val_ds.shard(num_shards=world_size, index=rank, contiguous=True)

#    model = ddp_model.module       # unwrap
    model.eval()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        local_results = evaluate_model_on_dataset(
            model, tokenizer, val_ds_rank,
            model_name = cfg.model_name,
            device     = f"cuda:{torch.cuda.current_device()}",
            lang       = cfg.lang,
            batch_size = cfg.eval_bs,
            temp       = cfg.eval_temp,
            at_k       = cfg.at_k,
            instruct_flag = True,
            with_reasoning = cfg.with_reasoning,
            verbose    = False
        )
    return local_results

        # compute pass@k, dump JSON, etc.


def linear_with_min_lr(step, total_steps, warmup_steps, min_factor=0.1):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(min_factor, 1.0 - progress * (1.0 - min_factor))
import math

def linear_with_floor_via_fake_T(actual_steps: int, min_factor: float, warmup_steps: int = 0):
    # returns a fake num_training_steps to pass to HF linear scheduler
    return max(actual_steps + 1, math.ceil((actual_steps - min_factor*warmup_steps) / (1.0 - min_factor)))
        


_STEP_RE = re.compile(r"(?:^|[_-])step[_-]?(\d+)\.pt$", re.IGNORECASE)

def _find_latest_pt(path: str) -> Optional[str]:
    """Return path to the *.pt file with the largest step number in a directory.
    Accepts names like: model_step_500.pt, myname_step_1234.pt, ..."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None

    best_file, best_step = None, -1
    for fname in os.listdir(path):
        if not fname.endswith(".pt"):
            continue
        m = _STEP_RE.search(fname)
        step = int(m.group(1)) if m else -1
        if step > best_step:
            best_step, best_file = step, fname
    return os.path.join(path, best_file) if best_file else None

def _maybe_strip_module(state_dict):
    """If keys start with 'module.', strip them."""
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def _maybe_add_module(state_dict):
    """If keys do NOT start with 'module.', add them. Useful if model is wrapped in DDP."""
    if not state_dict:
        return state_dict
    if all(not k.startswith("module.") for k in state_dict.keys()):
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint_if_any(
    resume_path: Optional[str],
    model,
    optimizer=None,
    scheduler=None,
    map_location=None,
    ddp_wrapped: bool = False,
    strict: bool = True,
) -> Tuple[int, Optional[str]]:
    """
    Returns:
      (start_step, loaded_path)
    start_step is 0 if nothing loaded.
    """
    if resume_path is None:
        return 0, None

    ckpt_path = _find_latest_pt(resume_path)
    if ckpt_path is None:
        print(f"[resume] WARNING: nothing found at {resume_path}")
        return 0, None

    print(f"[resume] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)

    saved_step = int(ckpt.get("step", 0))

    sd = ckpt.get("model_state_dict", {})
    if not ddp_wrapped:
        # loading into a non-DDP model
        sd = _maybe_strip_module(sd)
    else:
        # loading into a DDP-wrapped model (model.module is the real module)
        # We can load on model.module; state_dict keys in ckpt may or may not contain 'module.'
        target = model.module
        try:
            target.load_state_dict(_maybe_strip_module(sd), strict=strict)
        except RuntimeError:
            # try with module prefix
            target.load_state_dict(_maybe_add_module(sd), strict=strict)
    if not ddp_wrapped:
        try:
            model.load_state_dict(sd, strict=strict)
        except RuntimeError as e:
            # In case ckpt had "module." and earlier strip failed for some mismatch
            print(f"[resume] load_state_dict strict={strict} failed once, retrying with module prefix. Err: {e}")
            model.load_state_dict(_maybe_add_module(sd), strict=False)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[resume] WARNING: optimizer state not loaded cleanly: {e}")

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"[resume] WARNING: scheduler state not loaded cleanly: {e}")

    print(f"[resume] Loaded step={saved_step}")
    return saved_step, ckpt_path

def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)

def cfg_init(parser):
    cfg=parser.parse_args()

    if cfg.unfreeze_ids is None:
        cfg.unfreeze_ids = [[0]] * cfg.num_epochs
    elif type(cfg.unfreeze_ids[0])!=list:
        cfg.unfreeze_ids=[cfg.unfreeze_ids]                 
    cfg.num_epochs=max([cfg.num_epochs,len(cfg.unfreeze_ids)])
        
    if len(cfg.unfreeze_ids)<cfg.num_epochs:
        cfg.unfreeze_ids+=[cfg.unfreeze_ids[-1]]*(cfg.num_epochs-len(cfg.unfreeze_ids))
    if cfg.max_step_per_epoch is None:
        cfg.max_step_per_epoch = [1000] * cfg.num_epochs
    elif type(cfg.max_step_per_epoch)!=list:
        cfg.max_step_per_epoch=[cfg.max_step_per_epoch]                 
    if len(cfg.max_step_per_epoch)<cfg.num_epochs:
        cfg.max_step_per_epoch+=[cfg.max_step_per_epoch[-1]]*(cfg.num_epochs-len(cfg.max_step_per_epoch))
    if cfg.do_eval is None:
        cfg.do_eval=[False]*cfg.num_epochs
    else:
        if type(cfg.do_eval)!=list:
            cfg.do_eval=[cfg.do_eval]        
        cfg.do_eval+=[False]*(cfg.num_epochs-len(cfg.do_eval))
        
    return cfg
        
def main():
    parser = argparse.ArgumentParser()
    #  ─── model / data ─────────────────────────────
    model_name="Qwen/Qwen2.5-3B-Instruct"
    teacher_name="Qwen/Qwen2.5-Coder-14B-Instruct"
    parser.add_argument("--model_name",   default=model_name)
    parser.add_argument("--print_every",   type=int,default=10)
    parser.add_argument("--teacher_name",   default=teacher_name)
    parser.add_argument("--activate_rl", action="store_true")
    parser.add_argument("--reverse_kl", action="store_true")
    parser.add_argument("--kl_temp", type=float,default=1)
    parser.add_argument("--do_eval", type=ast.literal_eval,       default=None)
    parser.add_argument("--train_dataset",      default='codeforces_cot_filtered_truncated_v1')
    parser.add_argument("--val_dataset",      default='open-r1/codeforces')
    parser.add_argument("--checkpoint_every", type=int,      default=500)
    parser.add_argument("--lr_fac", type=float,      default=1)
    parser.add_argument("--split",        default="test")
    parser.add_argument("--keep_it_smooth",  action="store_true")
    parser.add_argument("--num_epochs",    type=int,    default=10)
    parser.add_argument("--unfreeze_ids",  type=ast.literal_eval,      default=None)
    parser.add_argument("--max_step_per_epoch",  type=ast.literal_eval, default=None)
    #  ─── eval specific ───────────────────────────
    parser.add_argument("--eval_bs",      type=int,   default=8)
#    parser.add_argument("--train_bs",      type=int,   default=8)
    parser.add_argument("--val_sample_per_gpu",      type=int,   default=8)
    parser.add_argument("--eval_temp",    type=float, default=0.001)
    parser.add_argument("--lang",         default="cpp")
    parser.add_argument("--at_k",         type=int,   default=1)
    parser.add_argument("--with_reasoning", action="store_true")
    parser.add_argument("--instruct_flag",  action="store_true")
    #  ─── misc ────────────────────────────────────
    parser.add_argument("--run_name",     default="run")
    parser.add_argument("--eval_every",   type=int, default=100)
    # add to your arg parser / cfg
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Path to a checkpoint .pt file OR a directory containing .pt files. "
                            "If a directory, the checkpoint with the largest *_step_XXXX.pt is loaded.")

    cfg= cfg_init(parser)   

    ACCESS_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

    #model_name="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    device='cuda'
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher_name,token=ACCESS_TOKEN,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",device_map={"": 1})
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name,token=ACCESS_TOKEN,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_auth_token=ACCESS_TOKEN)
    if tokenizer.pad_token_id is None:
        # Add a new unique padding token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#    model.resize_token_embeddings(len(tokenizer))
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)  
#     model.resize_token_embeddings(len(tokenizer))
    teacher.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)  
   
    model.train()
    teacher.eval()
    model.config.use_cache = False          # required with checkpointing
    model.enable_input_require_grads()
    # This is critical — enables grads on inputs so checkpointing can backprop
#    if gradient_checkpointing_enabled:
 #           from functools import partial
  #          notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
   #         torch.utils.checkpoint.checkpoint = notfailing_checkpoint
    #        model.gradient_checkpointing_enable()

    use_compile = False
    if use_compile:
        model = torch.compile(model)
        teacher=torch.compile(teacher)
    lah=1
    train_head=True
    train_embedder=False
    batch_size=B
    max_new_tokens=2048
    temp=1
    horizon=50
    total_examples_per_gpu= 2**12
   # optimize!
    weight_decay=0.01
    use_fused=True
    num_warmup_steps=0
    max_lr=2e-5#$4e-5*(total_batch_size/(2**20))*cfg.lr_fac
    min_factor=0.25
        
#    optim_groups=helpers.configure_optimizers(model,weight_decay,print_it=False)
#    optimizer=torch.optim.AdamW(optim_groups,lr=max_lr,fused=use_fused) 
        
#    model = DDP(model, device_ids=[ddp_local_rank],find_unused_parameters=True)
    
 
 
    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda"# if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
 #   destroy_process_group()
#    import sys;sys.exit()
    # resume (now that model/opt/sched exist)
#    for param in model.parameters():
 #       param.requires_grad = False
#    print(model.model.embed_tokens.weight.requires_grad)
  #  if train_head:
   #     model.module.lm_head.weight.requires_grad=True
   # if train_embedder:
    #    model.module.model.embed_tokens.weight.requires_grad=True
    #if unfreeze_layers>0:
     #   for layer in model.module.model.layers[-unfreeze_layers:]:
      #      for param in layer.parameters():
       #         param.requires_grad = True


    # prepare the dataset
    data_codeforces_cot = load_from_disk(cfg.train_dataset)
    val_ds_full = load_dataset(cfg.val_dataset, split="test").select(range(cfg.val_sample_per_gpu*ddp_world_size))
#load_from_disk("codeforces_cot_filtered")

    tokenized_dataset_train=data_codeforces_cot.select(range(total_examples_per_gpu*ddp_world_size)).map(helpers.tokenize_codeforce, batched=False,fn_kwargs={
            'context_length':2048,'tokenizer':tokenizer,'include_assistant':False})
#    tokenized_dataset_test=data_codeforces_test.select(range(2**7)).map(helpers.tokenize_codeforce, batched=False,fn_kwargs={
 #           'context_length':8092,'tokenizer':tokenizer})

    collate_fn_pad_partial=partial(helpers.collate_fn_pad,pad_token_id=tokenizer.pad_token_id)
    dataloader_pretrain = DataLoader(tokenized_dataset_train, batch_size=B, shuffle=True, collate_fn=collate_fn_pad_partial)
#    dataloader_pretest = DataLoader(tokenized_dataset_test, batch_size=B, shuffle=True, collate_fn=collate_fn_pad_partial)


    

    train_loader = DataLoader(
        dataloader_pretrain.dataset,
        batch_size=B,
        collate_fn=dataloader_pretrain.collate_fn,
    )
    print('gradient accumulation steps',gradient_accumulation_steps)
    total_steps = (len(train_loader)//gradient_accumulation_steps)* cfg.num_epochs # Assuming 3 epochs
    fake_T=linear_with_floor_via_fake_T(total_steps,min_factor)
    final_lr = max_lr * min_factor
#    if ddp_rank==0:
 #       import code;code.interact(local=locals())
        
    keep_opt_stat=False
    if cfg.resume_path is not None:
        optim_groups=helpers.configure_optimizers(model,weight_decay,print_it=False)
        optimizer=torch.optim.AdamW(optim_groups,lr=max_lr,fused=use_fused) 
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            lr_end=final_lr,
            power=1.0,   # linear
        )#    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=fake_T)
        keep_opt_stat=True
        print('schedulers last recorded epoch',scheduler.last_epoch)
    else:
        optimizer=None
        scheduler=None

    start_step, start_epoch, loaded_path = helpers.load_checkpoint_if_any(
        cfg.resume_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location='cuda',
        ddp_wrapped=False,
        strict=True,
    )
    raw_model=model
    print('-----------------------------',start_step,start_epoch,loaded_path)
    if start_step>=cfg.max_step_per_epoch[start_epoch]:#handle the case where the code crashed after finishing an epoch but before starting to write on the next one 
        start_step=0
        start_epoch+=1
        keep_opt_stat=False
#    helpers.pretrain_simerr(model,dataloader_train,optimizer,scheduler,device,tokenizer.pad_token_id,move_to_device=True,max_num_steps=max_num_steps,dataloader_test=dataloader_pretest,horizon=horizon,gradient_accumulation_steps=gradient_accumulation_steps,num_epochs=num_epochs,num_test_batches=30,print_per_batch=print_every)
    #raw_model = model.module #if ddp else model # always contains the "raw" unwrapped model

   # device='cpu'
 #   print('calling the trainer')
#    import code;code.interact(locals=locals())
    sanitized_model_name = sanitize_filename(cfg.model_name.split("/")[-1])
    
    accuracy=[]
#    import code;code.interact(local=locals())
    for epoch in range(start_epoch,cfg.num_epochs):
        unfreeze_ids=cfg.unfreeze_ids[epoch]
        for param in raw_model.parameters():
            param.requires_grad = False
    #    print(model.model.embed_tokens.weight.requires_grad)
        if train_head:
            raw_model.lm_head.weight.requires_grad=True
        if train_embedder:
            raw_model.model.embed_tokens.weight.requires_grad=True
        for idx in unfreeze_ids:
            if idx>0:
                for param in raw_model.model.layers[-idx].parameters():
                    param.requires_grad = True
        if not keep_opt_stat:
            adjusted_lr_fac=1.0
            if cfg.keep_it_smooth:
                adjusted_lr_fac=1/len(unfreeze_ids)*2
            optim_groups=helpers.configure_optimizers(raw_model,weight_decay,print_it=False)
            optimizer=torch.optim.AdamW(optim_groups,lr=max_lr*adjusted_lr_fac,fused=use_fused) 
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                lr_end=final_lr,
                power=1.0,   # linear
            )#    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=fake_T)
            if cfg.keep_it_smooth:
                scheduler.last_epoch=start_step-1
                scheduler.step()

            print('unfrozen layers',unfreeze_ids)
            print(start_step,start_epoch,'loaded_path',loaded_path)
            total_opt_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("total optimizable parameters",total_opt_param/1e9,'B')
#        model.gradient_checkpointing_enable()
        filename = f"{sanitized_model_name}/{cfg.train_dataset}/epoch_{epoch}"
        checkpoint_dir_=checkpoint_dir+'/'+filename
        print('checkpoint_dir',checkpoint_dir_)
        os.makedirs(checkpoint_dir_, exist_ok=True)
#        helpers.pretrain_simerr(model,train_loader,optimizer,scheduler,device,tokenizer.pad_token_id,tokenizer.eos_token_id,lah=lah,move_to_device=True,max_num_steps=cfg.max_step_per_epoch[epoch],dataloader_test=None,horizon=horizon,gradient_accumulation_steps=gradient_accumulation_steps,num_test_batches=30,print_per_batch=print_every,batch_size=batch_size,tokenizer=tokenizer,max_new_tokens=max_new_tokens,do_sample=True,temp=temp,ddp_rank=ddp_rank,ddp_world_size=ddp_world_size,writer=writer,checkpoint_dir=checkpoint_dir_,past_epoch_steps=sum(cfg.max_step_per_epoch[:epoch]),start_step=start_step,unfreeze_idx=unfreeze_ids[-1],checkpoint_every=cfg.checkpoint_every)
        teacher.eval()
        helpers.distill_simerr(model,teacher,train_loader,optimizer,scheduler,device,tokenizer.pad_token_id,tokenizer.eos_token_id,lah=lah,move_to_device=True,max_num_steps=cfg.max_step_per_epoch[epoch],dataloader_test=None,horizon=horizon,gradient_accumulation_steps=gradient_accumulation_steps,num_test_batches=30,print_every=cfg.print_every,batch_size=batch_size,tokenizer=tokenizer,max_new_tokens=max_new_tokens,do_sample=True,temp=temp,writer=writer,checkpoint_dir=checkpoint_dir_,past_epoch_steps=sum(cfg.max_step_per_epoch[:epoch]),start_step=start_step,unfreeze_idx=unfreeze_ids[-1],checkpoint_every=cfg.checkpoint_every,activate_rl=cfg.activate_rl,reverse_kl=cfg.reverse_kl)
        start_step=0
        keep_opt_stat=False
        do_eval=cfg.do_eval[epoch]

        if do_eval:
            local_results =distributed_eval(model, val_ds_full, tokenizer, cfg)  # full ds passed in
            all_results = gather_results(local_results)

            total = len(all_results)
            pass_at_k = defaultdict(int)

            for r in all_results:
                success_at = next((s["sample_id"] for s in r["samples"] if s["success"]), None)
                if success_at:
                    for k in range(success_at, cfg.at_k + 1):
                        pass_at_k[k] += 1

            print(f"\n📊 Evaluation Results on split={cfg.split} ({total} problems):")
            for k in range(1, cfg.at_k + 1):
                count = pass_at_k.get(k, 0)
                print(f"  ✅ Pass@{k}: {count}/{total} ({count / total * 100:.2f}%)")

            os.makedirs("results", exist_ok=True)
            with open(f"results/distill_eval_{cfg.lang}_atk{cfg.at_k}_n{total}.json", "w") as f:
                json.dump(all_results, f, indent=2)
#                total_successes= sum(pass_at_k.get(k, 0) for k in range(1, cfg.at_k + 1))
            total_successes= pass_at_k.get(cfg.at_k, 0)# for k in range(1, cfg.at_k + 1))
            print(f"Total successes so far: {total_successes}")
            accuracy.append(total_successes/total)
            series = {f"@{k}": pass_at_k.get(k, 0) / total for k in range(1, cfg.at_k + 1)}

            writer.add_scalars("Eval/Pass", series, epoch)                
            #writer.add_scalar("Acc/Eval", total_successes/total, epoch)

    print(accuracy)   
        
    

if __name__ == "__main__":
    main()