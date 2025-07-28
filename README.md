# sphinxnautics
cat <<'EOF' > README.md
# sphinxnautics-7b: Fine-Tuned Qwen2.5-Coder-7B for Competitive Programming

This repository hosts code and instructions for running and reproducing results with the `sphinxnautics-7b` model — a fine-tuned version of `Qwen2.5-Coder-7B-Instruct` on competitive programming problems using next-token and KL-based distillation strategies.

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/codedbythecharles/sphinxnautics.git
cd sphinxnautics

## 🧰 Requirements

```bash
pip install -r requirements.txt



##🔥 Run inference with vLLM

Serve the SFT model:
```bash
./launch_vllm.sh 12288 1 8000 0 "haj1r/sphinxnautics-7b"

```bash
Serve the KL‑distilled model:
./launch_vllm.sh 12288 1 8000 1 "haj1r/sphinxnautics-7b-kl-distilled"


