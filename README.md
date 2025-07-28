# sphinxnautics
cat <<'EOF' > README.md
# sphinxnautics-7b: Fine-Tuned Qwen2.5-Coder-7B for Competitive Programming

This repository hosts code and instructions for running and reproducing results with the `sphinxnautics-7b` model — a fine-tuned version of `Qwen2.5-Coder-7B-Instruct` on competitive programming problems using next-token and KL-based distillation strategies.

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


##🔥 Run inference with vLLM

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
You can run multiple servers on different ports and launch parallel evaluations with different --start_idx and --port values to speed up the process. This is useful when evaluating large numbers of problems (e.g. 1000+).

- At the end of evaluation, a .json summary file is automatically saved to the results/ directory. This file contains the full pass@k breakdown, including per-problem outputs and an overall summary.
