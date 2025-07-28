import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.utils.data import DataLoader
from functools import partial
from datasets import load_from_disk,load_dataset
from torch.distributed import init_process_group, destroy_process_group
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
import importlib.util
use_flash = importlib.util.find_spec("flash_attn") is not None

#torch.set_float32_matmul_precision('high')

#model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
#device = f'cuda:{0}'
total_batch_size = 2**20 # for a 7B model target 2M per GPT3 paper but we are fine-tuning so it can be less. 
B = 1 # micro batch size
T = 16384 # average sequence length
gradient_accumulation_steps = total_batch_size // (B * T * 1)

writer = SummaryWriter(log_dir="runs/exp8")
checkpoint_dir = "checkpoints2"
os.makedirs(checkpoint_dir, exist_ok=True)


def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)

def cfg_init(parser):
    cfg=parser.parse_args()

    if cfg.unfreeze_ids is None:
        cfg.unfreeze_ids = [[0]] * cfg.num_epochs
    elif type(cfg.unfreeze_ids[0])!=list:
        cfg.unfreeze_ids=[cfg.unfreeze_ids]                 
    cfg.num_epochs=max([cfg.num_epochs,len(cfg.unfreeze_ids)])
    print('----------------',cfg.unfreeze_ids,cfg.num_epochs)
    if len(cfg.unfreeze_ids)<cfg.num_epochs:
        cfg.unfreeze_ids+=[cfg.unfreeze_ids[-1]]*(cfg.num_epochs-len(cfg.unfreeze_ids))
    if cfg.max_step_per_epoch is None:
        cfg.max_step_per_epoch = [1000] * cfg.num_epochs
    elif type(cfg.max_step_per_epoch)!=list:
        cfg.max_step_per_epoch=[cfg.max_step_per_epoch]                 
    if len(cfg.max_step_per_epoch)<cfg.num_epochs:
        cfg.max_step_per_epoch+=[cfg.max_step_per_epoch[-1]]*(cfg.num_epochs-len(cfg.max_step_per_epoch))
    return cfg
        
def main():
    parser = argparse.ArgumentParser()
    #  ─── model / data ─────────────────────────────
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct"
    parser.add_argument("--model_name",   default=model_name)
    parser.add_argument("--train_dataset",      default='codeforces_cot_filtered_truncated_v1')
    parser.add_argument("--val_dataset",      default='open-r1/codeforces')
    parser.add_argument("--checkpoint_every", type=int,      default=500)
    parser.add_argument("--lr_fac", type=float,      default=1)
    parser.add_argument("--split",        default="test")
    parser.add_argument("--keep_it_smooth",  action="store_true")
    parser.add_argument("--with_torch_compile",  action="store_true")
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
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name,token=ACCESS_TOKEN ,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2" if use_flash else "eager",device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_auth_token=ACCESS_TOKEN )
    if tokenizer.pad_token_id is None:
        # Add a new unique padding token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
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

  #  device = f'cuda:{0}'
    device='cuda'
 #   torch.cuda.set_device(device)
#    model.to(device)
    use_compile = cfg.with_torch_compile
    if use_compile:
        model = torch.compile(model)
    lah=0
    train_head=True
    train_embedder=False
    batch_size=B
    max_new_tokens=8192
    temp=1
    horizon=50
    print_every=10
        
   # optimize!
    weight_decay=0.01
    use_fused=True
    num_warmup_steps=0
    max_lr=4e-5*(total_batch_size/(2**20))*cfg.lr_fac
    min_factor=0.25
        
#    optim_groups=helpers.configure_optimizers(model,weight_decay,print_it=False)
#    optimizer=torch.optim.AdamW(optim_groups,lr=max_lr,fused=use_fused) 
        
#    model = DDP(model, device_ids=[ddp_local_rank],find_unused_parameters=True)
    
 
 
    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

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
    val_ds_full = load_dataset(cfg.val_dataset, split="test").select(range(cfg.val_sample_per_gpu))
#load_from_disk("codeforces_cot_filtered")

    tokenized_dataset_train=data_codeforces_cot.select(range(2**10)).map(helpers.tokenize_codeforce, batched=False,fn_kwargs={
            'context_length':16384,'tokenizer':tokenizer})
#    tokenized_dataset_test=data_codeforces_test.select(range(2**7)).map(helpers.tokenize_codeforce, batched=False,fn_kwargs={
 #           'context_length':8092,'tokenizer':tokenizer})

    collate_fn_pad_partial=partial(helpers.collate_fn_pad,pad_token_id=tokenizer.pad_token_id)
    dataloader_pretrain = DataLoader(tokenized_dataset_train, batch_size=B, shuffle=True, collate_fn=collate_fn_pad_partial)
#    dataloader_pretest = DataLoader(tokenized_dataset_test, batch_size=B, shuffle=True, collate_fn=collate_fn_pad_partial)


    

    train_loader = DataLoader(
        dataloader_pretrain.dataset,
        batch_size=B,
#        sampler=sampler,
        collate_fn=dataloader_pretrain.collate_fn,
    )
    
 #       import code;code.interact(local=locals())
    sanitized_model_name = sanitize_filename(cfg.model_name.split("/")[-1])
    
    accuracy=[]
    do_eval=False
    start_epoch=0
#    import code;code.interact(local=locals())
    model.eval()
    for epoch in range(start_epoch,cfg.num_epochs):
        unfreeze_ids=cfg.unfreeze_ids[epoch]
        num_epochs_local=1
     #   helpers.pretrain_simerr(model,train_loader,optimizer,scheduler,device,tokenizer.pad_token_id,tokenizer.eos_token_id,lah=lah,move_to_device=True,max_num_steps=cfg.max_step_per_epoch[epoch],dataloader_test=None,horizon=horizon,gradient_accumulation_steps=gradient_accumulation_steps,num_epochs=num_epochs_local,num_test_batches=30,print_per_batch=print_every,batch_size=batch_size,tokenizer=tokenizer,max_new_tokens=max_new_tokens,do_sample=True,temp=temp,ddp_rank=ddp_rank,ddp_world_size=ddp_world_size,writer=writer,checkpoint_dir=checkpoint_dir_,past_epoch_steps=sum(cfg.max_step_per_epoch[:epoch]),start_step=start_step,unfreeze_idx=unfreeze_ids[-1],checkpoint_every=cfg.checkpoint_every)
        model.eval()
        print('device is',device)
        helpers.compute_logprobs(model,train_loader,device,tokenizer.pad_token_id,tokenizer.eos_token_id,lah=lah,move_to_device=True,max_num_steps=cfg.max_step_per_epoch[epoch],dataloader_test=None,horizon=horizon,gradient_accumulation_steps=gradient_accumulation_steps,num_epochs=num_epochs_local,num_test_batches=30,print_per_batch=print_every,batch_size=batch_size,tokenizer=tokenizer,max_new_tokens=max_new_tokens,do_sample=True,temp=temp)
        start_step=0
        keep_opt_stat=False
    

if __name__ == "__main__":
    main()