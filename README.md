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

To run evaluation with our framework:

```bash
python3 -m eval_codeforces \
  --at_k=16 \
  --temperature=0.7 \
  --port=8000 \
  --start_idx=0 \
  --max_problems=1000 \
  --lang=cpp \
  --split=test
```

💡 Note: 

- The --port flag allows you to point the evaluator to a specific vLLM server instance.
You can run multiple servers on different ports and launch parallel evaluations with different --start_idx and --port values to speed up the process. This is useful when evaluating large numbers of problems.

- At the end of evaluation, a .json summary file is automatically saved to the disk. This file contains the full pass@k breakdown, including per-problem outputs and an overall summary.

## 🏋️ Training
We provide two ways to train:

(1) Supervised Fine-Tuning (SFT): Trains a model directly on problem descriptions using next-token prediction. To reproduce our SFT model training, run:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train_model.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_dataset haj1r/sphinxnautics-codeforces-cot-v3 \
  --test_dataset open-r1/codeforces \
  --num_epochs 4 \
  --max_step_per_epoch [500,1000,1000,1000] \
  --unfreeze_ids [[1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,7,8],[1,2,3,4],[1,2,3,4]] \
  --init_max_CL 2048 \
  --instruct_flag \
  --with_reasoning \
  --eval_bs 1 \
  --val_sample_per_gpu 16 \
  --at_k 8 \
  --eval_temp 0.7 \
  --keep_it_smooth
```

(2) KL-Distillation: Fine-tunes a student model (e.g., Qwen2.5-Coder-7B-Instruct) using outputs from a larger teacher model (e.g., Qwen2.5-Coder-32B-Instruct) to guide learning via KL-divergence loss. Example:

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

