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
from torch.distributed import barrier, is_initialized
from torch.utils.tensorboard import SummaryWriter

from functools import partial
from datasets import load_from_disk,load_dataset
from collections import defaultdict
#torch.set_float32_matmul_precision('high')
import helpers
from hf_evaluation import evaluate_model_on_dataset
import importlib
from config_utils import load_config, select, _normalize_sft
from log_utils import init_logger
import yaml
from pathlib import Path
from argparse import Namespace, ArgumentParser

use_flash = importlib.util.find_spec("flash_attn") is not None

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if not torch.distributed.is_initialized():
    init_process_group(backend='nccl')
    
## setting up ddp ranks and devices
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
target_batch_tokens = 2**20 # for a 7B model target 2M per GPT3 paper but we are fine-tuning so it can be less. 
B = 1 #  batch size for dataloaders
T = 16384 # maximum sequence length


# utils/save_cfg.py

def _ns_to_dict(ns: Namespace):
    """Recursively turn argparse.Namespace / DotDict into plain dict."""
    out = {}
    for k, v in vars(ns).items():
        if isinstance(v, Namespace):
            out[k] = _ns_to_dict(v)
        else:
            out[k] = v
    return out

def save_cfg(cfg: Namespace, *paths):
    cfg_dict = _ns_to_dict(cfg)
    for p in paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".json":
            p.write_text(json.dumps(cfg_dict, indent=2))
        else:  # default YAML
            p.write_text(yaml.dump(cfg_dict, sort_keys=False))


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
            model_name = cfg.model.name,
            device     = f"cuda:{torch.cuda.current_device()}",
            lang       = cfg.lang,
            batch_size = cfg.eval.eval_bs,
            temp       = cfg.eval.temp,
            at_k       = cfg.eval.at_k,
            instruct_flag = True,
            with_reasoning = cfg.eval.with_reasoning,
            verbose    = False
        )
    return local_results

        # compute pass@k, dump JSON, etc.

# ─────────────────────────────────────────────────────────────
def wrapper(raw_model: torch.nn.Module, backend: str):
    if backend == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(raw_model, device_ids=[torch.cuda.current_device()])

    elif backend == "fsdp":
        from functools import partial
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

        # build the callable policy ← **partial(...)** is required
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e8)

        mp = MixedPrecision(param_dtype=torch.bfloat16,
                            reduce_dtype=torch.bfloat16,
                            buffer_dtype=torch.bfloat16)

        return FSDP(
            raw_model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp,
            use_orig_params=True,
            device_id=torch.cuda.current_device(),
            cpu_offload=CPUOffload(offload_params=False),
        )
    else:
        raise ValueError(f"Unknown backend {backend}")
# ─────────────────────────────────────────────────────────────
        

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
    """
    parser = argparse.ArgumentParser()
    
    #  ─── model / data ─────────────────────────────
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct"
    parser.add_argument("--model_name",   default=model_name)
    parser.add_argument("--verbose",action="store_true")
    parser.add_argument("--train_dataset",      default='haj1r/sphinxnautics-codeforces-cot-v3')
    parser.add_argument("--test_dataset",      default='open-r1/codeforces')
    parser.add_argument("--checkpoint_every", type=int,      default=500)
    parser.add_argument("--max_CL", type=int,      default=16384)
    parser.add_argument("--checkpoint_dir", type=str,      default='checkpoints')
    parser.add_argument("--experiment_id", type=int,      default=1)
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
    parser.add_argument("--eval_every",   type=int, default=100)
    # add to your arg parser / cfg
    parser.add_argument("--resume_path", type=str, default=None,
                        help="Path to a checkpoint .pt file OR a directory containing .pt files. "
                            "If a directory, the checkpoint with the largest *_step_XXXX.pt is loaded.")
    """
    cfg  = load_config()         # parses CLI
    logger = init_logger(cfg.logging.level,
                         cfg.experiment.output_dir,
                         rank=int(os.getenv("RANK", "0")))
    sft  = select(cfg, "sft")    # task block
    eval_cfg  = select(cfg, "eval")    # task block
    learning  = select(cfg, "learning")    # task block
    tokcfg =select(cfg, "tokenizer")
    _normalize_sft(sft)
    B=sft.micro_batch_size
    
    # saving configs to run and chekpoint folders for reproduciblity
    run_dir = Path(f"runs/exp{cfg.experiment.id}")
    ckpt_dir = Path(cfg.experiment.output_dir)
    save_cfg(cfg, run_dir/"config_used.yaml", ckpt_dir/"config_used.yaml")
    
#    print("▶ unfreeze_ids elem 0 type:", type(sft.unfreeze_ids),type(sft.unfreeze_ids[0][0]))
#    import code;code.interact(local=locals())

#    cfg= cfg_init(parser)   
#    if ddp_rank==0 and cfg.verbose:
 #       print('layer schedules for training',sft.unfreeze_ids)# each epoch will train the layers specified in the list. The entries are distance from the last layer so [0,1] means layer[-1],layer[-2] will be trained 
    gradient_accumulation_steps = sft.total_batch_tokens // (B * T * ddp_world_size)

    logger.info(f'Layer schedules for training: {sft.unfreeze_ids}')
    logger.info(f'Gradient accumulation steps (calculated): {gradient_accumulation_steps}')
    
    writer = SummaryWriter(log_dir=f"runs/exp{cfg.experiment.id}")
    checkpoint_dir = cfg.experiment.output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    map_loc = f"cuda:{ddp_rank}"# if device_type == "cuda" else "cpu"
    ACCESS_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
    #model_name="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name,token=ACCESS_TOKEN,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2" if use_flash else "eager")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, token=ACCESS_TOKEN)
    if tokenizer.pad_token_id is None:
        # Add a new unique padding token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)  
    model.train()
    model.config.use_cache = False          # required with checkpointing
    if sft.enable_grad_chkpt:
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
    total_examples_per_gpu=2**13
    temp=1
    horizon=50
    print_every=10
        
   # optimize!
    weight_decay=learning.weight_decay
    use_fused=True
    num_warmup_steps=0
    max_lr=float(learning.max_lr)*(sft.total_batch_tokens/(target_batch_tokens))*learning.lr_init_adjust_fac
    logger.info(f'max learning rate: {max_lr}')
    final_lr=max_lr*learning.lr_final_decay_fac
        

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)


    # prepare the dataset for train/val loss
    data_codeforces_cot_train = load_dataset(cfg.train_dataset,split='train')#['train']load_from_disk(cfg.train_dataset,split='train')
    data_codeforces_cot_val = load_dataset(cfg.train_dataset,split='validation')
    #prepaer the dataset for end to end testing
    
    test_ds_full = load_dataset(eval_cfg.dataset, split="test").select(range(sft.val_sample_per_gpu*ddp_world_size))
    logger.info(f"full ds len {len(test_ds_full)} ddp_rank {ddp_rank} ddp_worldsize {ddp_world_size}")    

#load_from_disk("codeforces_cot_filtered")
    
    """
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
    """
    
    keep_opt_stat=False
    total_steps=(total_examples_per_gpu//gradient_accumulation_steps)* sft.num_epochs
    if cfg.resume_path is not None:
        # 2. ALWAYS create the optimizer & scheduler ------------
        optim_groups=helpers.configure_optimizers(model,weight_decay)
        optimizer=torch.optim.AdamW(optim_groups,lr=max_lr,fused=use_fused) 
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            lr_end=final_lr,
            power=1.0,   # linear
        )

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
    if start_step>=sft.max_step_per_epoch[start_epoch]:#handle the case where the code crashed after finishing an epoch but before starting to write on the next one 
        start_step=0
        start_epoch+=1
        keep_opt_stat=False
#    helpers.pretrain_simerr(model,dataloader_train,optimizer,scheduler,device,tokenizer.pad_token_id,move_to_device=True,max_num_steps=max_num_steps,dataloader_test=dataloader_pretest,horizon=horizon,gradient_accumulation_steps=gradient_accumulation_steps,num_epochs=num_epochs,num_test_batches=30,print_per_batch=print_every)
    #raw_model = model.module #if ddp else model # always contains the "raw" unwrapped model

   # device='cpu'
 #   print('calling the trainer')
#    import code;code.interact(locals=locals())
    sanitized_model_name = sanitize_filename(cfg.model.name.split("/")[-1])
    
    accuracy=[]
#    import code;code.interact(local=locals())
    for epoch in range(start_epoch,sft.num_epochs):
        print("▶ unfreeze_ids type:", type(sft.unfreeze_ids),'layers',sft.unfreeze_ids)
        total = total_examples_per_gpu * ddp_world_size
        
        max_size = len(data_codeforces_cot_train)

        # Cap the selection to available size
        num_samples = min(total, max_size)

        tokenized_dataset_train=data_codeforces_cot_train.select(range(num_samples)).map(helpers.tokenize_codeforce, batched=False,fn_kwargs={
            'context_length':min([int(tokcfg.init_max_CL*2**epoch),tokcfg.max_CL]),'tokenizer':tokenizer,'truncation':tokcfg.truncation,'padding':tokcfg.padding})
        tokenized_dataset_val=data_codeforces_cot_val.map(helpers.tokenize_codeforce, batched=False,fn_kwargs={
            'context_length':min([int(tokcfg.init_max_CL*2**epoch),tokcfg.max_CL]),'tokenizer':tokenizer,'truncation':tokcfg.truncation,'padding':tokcfg.padding})
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
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataloader_preval.dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            dataloader_preval.dataset,
            batch_size=B,
            sampler=val_sampler,
            collate_fn=dataloader_preval.collate_fn,
        )
        
        logger.info(f"number of training iterations in process {ddp_rank}:{len(train_loader),len(sampler)}")
        logger.info(f"number of validation iterations in process {ddp_rank}:{len(val_loader),len(val_sampler)}")
        
        
        
        
        unfreeze_ids=sft.unfreeze_ids[epoch]
        if unfreeze_ids:            
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
            num_trainable_layers=len(unfreeze_ids)
        else:
            num_trainable_layers=len(raw_model.model.layers)
            unfreeze_ids=list(range(-1, -num_trainable_layers - 1, -1))
        if not keep_opt_stat:
            adjusted_lr_fac=1.0
            if learning.keep_it_smooth:
                adjusted_lr_fac=1/num_trainable_layers*2
            optim_groups=helpers.configure_optimizers(raw_model,weight_decay)
            optimizer=torch.optim.AdamW(optim_groups,lr=max([max_lr*adjusted_lr_fac,final_lr*1.1]),fused=use_fused) 
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                lr_end=final_lr,
                power=1.0,   # linear
            )#    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=fake_T)
            if learning.keep_it_smooth:
                scheduler.last_epoch=start_step-1
                scheduler.step()

        if ddp_rank==0:
            logger.info(f'unfrozen layers {unfreeze_ids}')
            logger.info(f'start step {start_step} start epoch {start_epoch} loaded_path {loaded_path}')
            total_opt_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"total optimizable parameters {total_opt_param/1e9} B")

        logger.info(f"backend: {cfg.dist_backend}")
        model = wrapper(raw_model, cfg.dist_backend)
        raw_model = model.module if cfg.dist_backend=='ddp' else model
        ckpt_style= 'auto' if cfg.dist_backend=='ddp' else 'full'
#        ckpt_style= 'auto'
        if sft.enable_grad_chkpt:
            raw_model.gradient_checkpointing_enable()
        filename = f"{sanitized_model_name}/{cfg.train_dataset}/epoch_{epoch}"
        checkpoint_dir_=checkpoint_dir+'/'+filename
        if ddp_rank==0:
            os.makedirs(checkpoint_dir_, exist_ok=True)
        helpers.pretrain_simerr(model,train_loader,optimizer,scheduler,device,tokenizer.pad_token_id,tokenizer.eos_token_id,dataloader_test=val_loader,lah=lah,move_to_device=True,max_num_steps=sft.max_step_per_epoch[epoch],horizon=horizon,gradient_accumulation_steps=gradient_accumulation_steps,num_test_batches=30,print_per_batch=print_every,batch_size=batch_size,tokenizer=tokenizer,max_new_tokens=max_new_tokens,do_sample=True,temp=temp,ddp_rank=ddp_rank,ddp_world_size=ddp_world_size,writer=writer,checkpoint_dir=checkpoint_dir_,past_epoch_steps=sum(sft.max_step_per_epoch[:epoch]),start_step=start_step,unfreeze_idx=unfreeze_ids[-1],checkpoint_every=sft.checkpoint_every,ckpt_style=ckpt_style,dist_backend=cfg.dist_backend,eval_every=sft.eval_every)
        logger.info('epoch ended...')
#        raw_model=model.module
        raw_model = model.module if cfg.dist_backend == "ddp" else model
        start_step=0
        keep_opt_stat=False
        do_eval=sft.do_eval[epoch]
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

                    logger.info(f"\n📊 Evaluation Results on split={cfg.split} ({total} problems):")
                    for k in range(1, cfg.at_k + 1):
                        count = pass_at_k.get(k, 0)
                        print(f"  ✅ Pass@{k}: {count}/{total} ({count / total * 100:.2f}%)")

                    os.makedirs("results", exist_ok=True)
                    with open(f"results/ddp_eval_{cfg.lang}_atk{cfg.at_k}_n{total}.json", "w") as f:
                        json.dump(all_results, f, indent=2)
    #                total_successes= sum(pass_at_k.get(k, 0) for k in range(1, cfg.at_k + 1))
                    total_successes= pass_at_k.get(cfg.at_k, 0)# for k in range(1, cfg.at_k + 1))
                    logger.info(f"Total successes so far: {total_successes}")
                    accuracy.append(total_successes/total)
                    series = {f"@{k}": pass_at_k.get(k, 0) / total for k in range(1, cfg.at_k + 1)}

                    writer.add_scalars("Eval/Pass", series, epoch)                
                    #writer.add_scalar("Acc/Eval", total_successes/total, epoch)

    logger.info(accuracy)   
        
    destroy_process_group()
    

if __name__ == "__main__":
    main()