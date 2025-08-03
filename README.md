# sphinxnautics: Fine-Tuning and Evaluating Qwen2.5-7B-Instruct on Codeforces

This repository hosts code and instructions for running and reproducing results with the `sphinxnautics-7b` model — a fine-tuned version of `Qwen2.5-7B-Instruct` on competitive programming problems using SFT and KL-based distillation strategies.

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/codedbythecharles/sphinxnautics.git
cd sphinxnautics
```

## 🧰 Requirements

```bash
pip install -r requirements.txt
```


## 🔥 Run inference with vLLM

Serve the SFT model:
```bash
./launch_vllm.sh 12288 1 8000 0 "haj1r/sphinxnautics-7b"
```


Serve the KL‑distilled model:

```bash
./launch_vllm.sh 12288 1 8000 1 "haj1r/sphinxnautics-7b-kl-distilled"
```


## 🧪 Evaluation on Codeforces (Pass@K)
We evaluate using the open‑r1/codeforces dataset.
This dataset does not include any code — only problem specifications (title, description, input/output format, examples).
Languages supported: cpp, python.

The evaluation script reads sensible defaults from configs/config.yaml. The eval section in that file defines fields such as at_k, split, port and temp (sampling temperature). To run evaluation with our framework, 

```bash
python3 -m eval_codeforces \
  --eval.port=8000 \
```

💡 Note: 

- The --port flag allows you to point the evaluator to a specific vLLM server instance.
You can run multiple servers on different ports and launch parallel evaluations with different --start_idx and --port values to speed up the process. This is useful when evaluating large numbers of problems.

- At the end of evaluation, a .json summary file is automatically saved to the disk. This file contains the full pass@k breakdown, including per-problem outputs and an overall summary.

- Any field in the eval section can be overridden at runtime using dotted CLI flags. For example, to evaluate pass@16 at temperature 0.7 on the test split for 100 problems starting at index 0, run:

```bash
python3 -m eval_codeforces \
  --eval.at_k=16 \
  --eval.temperature=0.7 \
  --eval.port=8000 \
  --eval.start_idx=0 \
  --eval.max_problems=100 \
  --eval.dataset=open‑r1/codeforces\
  --eval.split=test
```


## 🏋️ Training
We provide two ways to train:

(1) Supervised Fine-Tuning (SFT): Trains a model directly on problem descriptions using next-token prediction. All hyper-parameters are in configs/config.yaml.  Override any of them with dotted CLI flags, e.g. `--model.name=Qwen/Qwen2.5-7B-Instruct`. You can also swap data-parallel back-ends (ddp vs fsdp) or pick which decoder layers to train per epoch.

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 # mitigate CUDA fragmentation

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_model.py \
  --dist_backend=ddp
```

FSDP is less prone to memory fragmentation issues:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_model.py \
  --dist_backend=fsdp \
  --sft.unfreeze_ids="[[1,2,3,4],[1,2,3]]"
```

(2) KL-Distillation: Fine-tunes a student model (e.g., Qwen2.5-7B-Instruct) using outputs from a larger teacher model (e.g., Qwen2.5-Coder-32B-Instruct) to guide learning via KL-divergence loss. Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_model_distill.py \
  --model_name CUDA_VISIBLE_DEVICES=0,1 python train_model_distill.py \
  --model_name Qwen/Qwen2.5-7B-Instruct\
  --teacher_name Qwen/Qwen2.5-Coder-32B-Instruct \
  --train_dataset <your_dataset> \
  --instruct_flag \
  --with_reasoning \
  --num_epochs 1 \
  --max_step_per_epoch 2000 \
  --checkpoint_every 500 \
  --keep_it_smooth
  --experiment_id 1
```

