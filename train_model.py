import os
import re
import ast
import math
import json
import argparse
from collections import defaultdict
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
import torch
assert torch.cuda.is_available()
from torch.optim import AdamW
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from functools import partial
from datasets import load_from_disk,load_dataset
from collections import defaultdict
#torch.set_float32_matmul_precision('high')
import helpers
from hf_evaluation import evaluate_model_on_dataset

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if not torch.distributed.is_initialized():
    init_process_group(backend='nccl')
    
## setting up ddp ranks and devices
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
total_batch_size = 2**20 # for a 7B model target 2M per GPT3 paper but we are fine-tuning so it can be less. 
B = 1 #  batch size for dataloaders
T = 16384 # maximum sequence length
gradient_accumulation_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {gradient_accumulation_steps}")

writer = SummaryWriter(log_dir="runs/exp16")
checkpoint_dir = "checkpoints5"
os.makedirs(checkpoint_dir, exist_ok=True)


# ---------- helpers ----------------------------------------------------------
def init_distributed():
#    ddp = int(os.environ.get("RANK", "-1")) != -1
 #   if ddp and not dist.is_initialized():
  #      dist.init_process_group("nccl")
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

def distributed_eval(ddp_model, val_ds, tokenizer, cfg):
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # 1. dataset slice for *this* rank
    val_ds_rank = val_ds.shard(num_shards=world_size, index=rank, contiguous=True)
    print('shared data prepared for eval')
    model = ddp_model.module       # unwrap
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
    if cfg.init_max_CL is None:
        cfg.init_max_CL=16384
            
    return cfg
        
def main():
    parser = argparse.ArgumentParser()
    #  ─── model / data ─────────────────────────────
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct"
    parser.add_argument("--model_name",   default=model_name)
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--train_dataset",      default='haj1r/sphinxnautics-codeforces-cot-v3')
 #   parser.add_argument("--val_dataset",      default='codeforces_cot_filtered_truncated_matched_val_v1')
    parser.add_argument("--test_dataset",      default='open-r1/codeforces')
    parser.add_argument("--checkpoint_every", type=int,      default=500)
    parser.add_argument("--lr_fac", type=float,      default=1)
    parser.add_argument("--init_max_CL", type=int,      default=None)
    parser.add_argument("--split",        default="test")
    parser.add_argument("--do_eval", type=ast.literal_eval,       default=None)
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
    if ddp_rank==0 and cfg.verbose:
        print('layer schedules for training',cfg.unfreeze_ids)# each epoch will train the layers specified in the list. The entries are distance from the last layer so [0,1] means layer[-1],layer[-2] will be trained 

    map_loc = f"cuda:{ddp_rank}"# if device_type == "cuda" else "cpu"
    ACCESS_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
    #model_name="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name,token=ACCESS_TOKEN,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=ACCESS_TOKEN)
    if tokenizer.pad_token_id is None:
        # Add a new unique padding token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)  
    model.train()
    model.config.use_cache = False          # required with checkpointing
    model.enable_input_require_grads()
    # This is critical — enables grads on inputs so checkpointing can backprop
#    if gradient_checkpointing_enabled:
 #           from functools import partial
  #          notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
   #         torch.utils.checkpoint.checkpoint = notfailing_checkpoint
    #        model.gradient_checkpointing_enable()

    device = f'cuda:{ddp_local_rank}'

    torch.cuda.set_device(device)
    model.to(device)
    use_compile = False
    if use_compile:
        model = torch.compile(model)
    lah=0
    train_head=True
    train_embedder=False
    batch_size=B
    max_new_tokens=8192
    total_examples_per_gpu=2**11
    temp=1
    horizon=50
    print_every=10
        
   # optimize!
    weight_decay=0.01
    use_fused=True
    num_warmup_steps=0
    max_lr=4e-5*(total_batch_size/(2**20))*cfg.lr_fac
    if cfg.verbose and master_process:
        print('max learning rate',max_lr)
    min_factor=0.25
        

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # prepare the dataset for train/val loss
    data_codeforces_cot_train = load_dataset(cfg.train_dataset,split='train')#['train']load_from_disk(cfg.train_dataset,split='train')
    data_codeforces_cot_val = load_dataset(cfg.train_dataset,split='validation')
    #prepaer the dataset for end to end testing
    test_ds_full = load_dataset(cfg.test_dataset, split="test").select(range(cfg.val_sample_per_gpu*ddp_world_size))
    if cfg.verbose:
        print('---------------------full ds rank',len(test_ds_full),'ddp local',ddp_rank,'ddp_worldsize',ddp_world_size)
#load_from_disk("codeforces_cot_filtered")
    final_lr = max_lr * min_factor

    keep_opt_stat=False
    total_steps=(total_examples_per_gpu//gradient_accumulation_steps)* cfg.num_epochs
    if cfg.resume_path is not None:
        optim_groups=helpers.configure_optimizers(model,weight_decay,print_it=False)
        optimizer=torch.optim.AdamW(optim_groups,lr=max_lr,fused=use_fused) 
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            lr_end=final_lr,
            power=1.0,   # linear
        )
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
        map_location=map_loc,
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
        tokenized_dataset_train=data_codeforces_cot_train.select(range(total_examples_per_gpu*ddp_world_size)).map(helpers.tokenize_codeforce, batched=False,fn_kwargs={
            'context_length':min([int(cfg.init_max_CL*2**epoch),16384]),'tokenizer':tokenizer,'truncation':True,'padding':'max_length'})
        tokenized_dataset_val=data_codeforces_cot_val.map(helpers.tokenize_codeforce, batched=False,fn_kwargs={
            'context_length':min([int(cfg.init_max_CL*2**epoch),16384]),'tokenizer':tokenizer,'truncation':True,'padding':'max_length'})
#    tokenized_dataset_test=data_codeforces_test.select(range(2**7)).map(helpers.tokenize_codeforce, batched=False,fn_kwargs={
 #           'context_length':8092,'tokenizer':tokenizer})
        collate_fn_pad_partial=partial(helpers.collate_fn_pad,pad_token_id=tokenizer.pad_token_id)
        dataloader_pretrain = DataLoader(tokenized_dataset_train, batch_size=B, shuffle=True, collate_fn=collate_fn_pad_partial)
        dataloader_preval = DataLoader(tokenized_dataset_val, batch_size=B, shuffle=True, collate_fn=collate_fn_pad_partial)
#    dataloader_pretest = DataLoader(tokenized_dataset_test, batch_size=B, shuffle=True, collate_fn=collate_fn_pad_partial)
        # DataLoader: important to use DistributedSampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataloader_pretrain.dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=True
        )
        train_loader = DataLoader(
            dataloader_pretrain.dataset,
            batch_size=B,
            sampler=sampler,
            collate_fn=dataloader_pretrain.collate_fn,
        )

        val_loader = DataLoader(
            dataloader_preval.dataset,
            batch_size=B,
            sampler=sampler,
            collate_fn=dataloader_preval.collate_fn,
        )
        
        if cfg.verbose:      
            print(f"number of training iterations in process {ddp_rank}:{len(train_loader),len(sampler)}")

        
        
        
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
            optimizer=torch.optim.AdamW(optim_groups,lr=max([max_lr*adjusted_lr_fac,final_lr*1.1]),fused=use_fused) 
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

        if ddp_rank==0:
            print('unfrozen layers',unfreeze_ids)
            print(start_step,start_epoch,loaded_path)
            total_opt_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("total optimizable parameters",total_opt_param/1e9,'B')
        model=DDP(raw_model,device_ids=[ddp_rank])
        model.module.gradient_checkpointing_enable()
        filename = f"{sanitized_model_name}/{cfg.train_dataset}/epoch_{epoch}"
        checkpoint_dir_=checkpoint_dir+'/'+filename
        if ddp_rank==0:
            os.makedirs(checkpoint_dir_, exist_ok=True)
        helpers.pretrain_simerr(model,train_loader,optimizer,scheduler,device,tokenizer.pad_token_id,tokenizer.eos_token_id,dataloader_test=val_loader,lah=lah,move_to_device=True,max_num_steps=cfg.max_step_per_epoch[epoch],dataloader_test=None,horizon=horizon,gradient_accumulation_steps=gradient_accumulation_steps,num_test_batches=30,print_per_batch=print_every,batch_size=batch_size,tokenizer=tokenizer,max_new_tokens=max_new_tokens,do_sample=True,temp=temp,ddp_rank=ddp_rank,ddp_world_size=ddp_world_size,writer=writer,checkpoint_dir=checkpoint_dir_,past_epoch_steps=sum(cfg.max_step_per_epoch[:epoch]),start_step=start_step,unfreeze_idx=unfreeze_ids[-1],checkpoint_every=cfg.checkpoint_every,eval_every=cfg.eval_every,verbose=cfg.verbose)
        if ddp_rank==0:
            print('epoch ended...')
        raw_model=model.module
        start_step=0
        keep_opt_stat=False
        do_eval=cfg.do_eval[epoch]
        if do_eval:
            local_results =distributed_eval(model, test_ds_full, tokenizer, cfg)  # full ds passed in
            all_results = gather_results(local_results)

            if ddp_rank == 0 and all_results is not None:
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
                    with open(f"results/ddp_eval_{cfg.lang}_atk{cfg.at_k}_n{total}.json", "w") as f:
                        json.dump(all_results, f, indent=2)
    #                total_successes= sum(pass_at_k.get(k, 0) for k in range(1, cfg.at_k + 1))
                    total_successes= pass_at_k.get(cfg.at_k, 0)# for k in range(1, cfg.at_k + 1))
                    print(f"Total successes so far: {total_successes}")
                    accuracy.append(total_successes/total)
                    series = {f"@{k}": pass_at_k.get(k, 0) / total for k in range(1, cfg.at_k + 1)}

                    writer.add_scalars("Eval/Pass", series, epoch)                
                    #writer.add_scalar("Acc/Eval", total_successes/total, epoch)

    print(accuracy)   
        
    destroy_process_group()
    

if __name__ == "__main__":
    main()