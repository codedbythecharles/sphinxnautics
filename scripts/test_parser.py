import argparse
import ast
def cfg_init(parser):
    cfg=parser.parse_args()
    if cfg.unfreeze_ids is None:
        cfg.unfreeze_ids = [[]] * cfg.num_epochs
    elif type(cfg.unfreeze_ids[0])!=list:
        cfg.unfreeze_ids=[cfg.unfreeze_ids]                 
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
    parser.add_argument("--split",        default="test")
    parser.add_argument("--num_epochs",    type=int,    default=10)
    parser.add_argument("--max_step_per_epoch",    type=int,    default=10)
    parser.add_argument("--unfreeze_ids",  type=ast.literal_eval,      default=None)
    #  ─── eval specific ───────────────────────────
    parser.add_argument("--eval_bs",      type=int,   default=8)
#    parser.add_argument("--train_bs",      type=int,   default=8)
    parser.add_argument("--val_sample_per_gpu",      type=int,   default=8)
    parser.add_argument("--eval_temp",    type=float, default=0.001)
    parser.add_argument("--lang",         default="cpp")
    parser.add_argument("--keep_it_smooth",  action="store_true")

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
    print('?',cfg.keep_it_smooth)
#cfg= parser.parse_args()    
    # Now safe to use args.num_epochs
    if cfg.unfreeze_ids is None:
        cfg.unfreeze_ids = [[]] * cfg.num_epochs
    elif type(cfg.unfreeze_ids[0])!=list:
        cfg.unfreeze_ids=[cfg.unfreeze_ids]
                 
    if len(cfg.unfreeze_ids)<cfg.num_epochs:
        cfg.unfreeze_ids+=[cfg.unfreeze_ids[-1]]*(cfg.num_epochs-len(cfg.unfreeze_ids))
    print(cfg.unfreeze_ids)
if __name__=='__main__':
    main()
    