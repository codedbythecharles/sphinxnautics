import json
import requests
from tqdm import tqdm
import subprocess
import tempfile
import os
import re
import helpers
import argparse
from collections import defaultdict
from datasets import load_dataset
#codeforces_api_key=10453d6076515f7adb99e27db0745e5eb0cb0188
#codeforces secret=0945da1bba8a2acdca66f9e0d0c0b6c0c33bf601
# Configs
LANGUAGE_ID = 54  # C++ (or use 71 for Python 3, etc.)

#VLLM_ENDPOINT = "http://localhost:8002/v1/chat/completions"  # Change port for different GPUs
#MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#MODEL_NAME ="Qwen/Qwen2.5-Coder-7B-Instruct"
#MODEL_NAME ="Qwen/Qwen2.5-Coder-32B-Instruct"
#MODEL_NAME = "meta-llama/Meta-Llama-3.3-70B-Instruct"
#MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
#MODEL_NAME ="meta-llama/CodeLlama-7b-Python-hf"
#MODEL_NAME ="meta-llama/CodeLlama-34b-Python-hf"
#MODEL_NAME ="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
#MODEL_NAME= "deepseek-ai/deepseek-coder-6.7b-instruct"
#MODE_NAME= "deepseek-ai/deepseek-coder-33b-instruct"
#MODE_NAME= "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
#MODE_NAME= "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
#Qwen/Qwen2.5-7B-Instruct
#"Qwen/Qwen-7B-Instruct"


def get_model_name_from_vllm(port: int) -> str:
    try:
        resp = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception as e:
        print(f"⚠️ Failed to get model name from VLLM on port {port}: {e}")
    return "unknown"




def load_problems(split='test',dataset_name='open-r1/codeforces'):
#    dataset = load_dataset("Qwen/CodeElo", split=split)
    dataset = load_dataset(dataset_name, split=split)
    return list(dataset)  # Convert to list of dicts

import math
from openai import OpenAI


import requests
import math

def query_model_new(prompt: dict, k: int, temperature: float = 1.0,
                     endpoint: str = None, return_logprobs: bool = False,
                     top_logprobs: int = 0, return_stats: bool = False):
    """
    prompt: {"system": "...", "user": "..."}
    Returns:
      - if return_logprobs == False:
          completions : List[str]
      - if return_logprobs == True:
          (completions,
           token_logprobs,     # List[List[Tuple[token:str, logprob:float]]], assistant-only
           stats)              # Optional: per-choice dicts with totals/ppl if return_stats
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user",   "content": prompt["user"]},
        ],
        "temperature": temperature,
        "repetition_penalty": 1.1,
        "max_tokens": 8192,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 20,
        "n": k,
    }
    print('we aere here with',return_logprobs)
    if return_logprobs:
        payload["logprobs"] = True          # return assistant token logprobs
        payload["top_logprobs"] = int(top_logprobs)
        # DO NOT set "prompt_logprobs": we only want assistant portion

    resp = requests.post(endpoint, json=payload)
    resp.raise_for_status()
    result = resp.json()

    completions = [c["message"]["content"] for c in result["choices"]]

    if not return_logprobs:
        return completions

    token_logprobs = []
    stats = [] if return_stats else None

    for c in result["choices"]:
        lp_obj = c.get("logprobs") or {}
        content = lp_obj.get("content") or []   # assistant tokens only
        seq = []
        total_lp = 0.0
        n_tok = 0
        for item in content:
            tok = item.get("token")
            lp  = item.get("logprob")
            if tok is None or lp is None:
                continue
            lp = float(lp)
            seq.append((tok, lp))
            total_lp += lp
            n_tok += 1
        token_logprobs.append(seq)

        if return_stats:
            avg_nll = (-total_lp / n_tok) if n_tok > 0 else float("inf")
            ppl = math.exp(avg_nll) if n_tok > 0 else float("inf")
            stats.append({
                "num_tokens": n_tok,
                "total_logprob": total_lp,   # sum log p(token)
                "avg_nll": avg_nll,          # mean negative log-prob
                "perplexity": ppl,
            })

    return (completions, token_logprobs, stats) if return_stats else (completions, token_logprobs)


def query_model(prompt: str, k: int,temperature:float=1.0, endpoint: str = None) -> list[str]:
    print('query is called with',k,'model_name',MODEL_NAME,'temp',temperature)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": prompt['user']}
        ],
        "temperature": temperature,
        "repetition_penalty": 1.1,
        "max_tokens": 8192,#12288,
        "do_sample":True,
        "top_p":0.8,
        "top_k":20,
        "n": k  # <-- IMPORTANT: request k completions
    }
#    print('len if payload',len(payload))
    response = requests.post(endpoint, json=payload)

#    response = requests.post(VLLM_ENDPOINT, json=payload)
    response.raise_for_status()
#    print('returned',response.json()["choices"])
    result = response.json()

    # Always return a list of `k` code completions
    completions = [choice["message"]["content"] for choice in result["choices"]]
#    if return_logprobs:        
 #      lps  = response.choices[0].logprobs.token_logprobs
#    print('completions are',len(completions),completions)
    return completions
#    return [choice["message"]["content"] for choice in response.json()["choices"]]


   


def main():
    print('hello!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--at_k', type=int, default=1, help='Number of completions to use for pass@k evaluation')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--lang', type=str, default='cpp')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='open-r1/codeforces')
    #parser.add_argument('--ports', type=lambda s: list(map(int, s.split(','))), default=[8000])
    parser.add_argument('--port', type=str, default="8000", help='Comma-separated list of ports for vLLM servers')
    parser.add_argument('--max_problems', type=int, default=10)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature for model generation')
    parser.add_argument('--with_reasoning', dest='with_reasoning', action='store_true', help='Include reasoning in the prompt')
    parser.add_argument('--no_reasoning', dest='with_reasoning', action='store_false', help='Exclude reasoning in the prompt')
    parser.set_defaults(with_reasoning=True)

    args = parser.parse_args()
    problems = load_problems(args.split,args.dataset)
    dataset_name=args.dataset.split('/')[-1]
    port_list = [int(p.strip()) for p in args.port.split(",")]
    print(f"🔧 Using ports: {port_list}")
    print(f"🔧 Language: {args.lang}")
    problems = problems[args.start_idx:args.start_idx+args.max_problems]

    global MODEL_NAME
    MODEL_NAME = args.model_name or get_model_name_from_vllm(port_list[0])
    print(f"🔧 Using model: {MODEL_NAME} Dataset: {args.dataset}")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import itertools

    port_cycle = itertools.cycle(port_list)
    def make_model(port):
        print('port is',port)
        endpoint = f"http://localhost:{port}/v1/chat/completions"
        return lambda prompt, k=args.at_k, temperature=args.temperature: query_model(
            prompt, k=k, temperature=temperature, endpoint=endpoint)

    def evaluate_single(problem, port):
        try:
            model = make_model(port)
            return helpers.evaluate_problem(problem, model, with_reasoning=args.with_reasoning, k=args.at_k, temperature=args.temperature,lang=args.lang)
        except Exception as e:
            print(f"❌ Error evaluating problem {problem.get('id', '???')} on port {port}: {e}")
            return None

    results = []
    with ThreadPoolExecutor(max_workers=len(port_list)) as executor:
        futures = {
            executor.submit(evaluate_single, prob, next(port_cycle)): prob
            for prob in problems
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                results.append(result)

    # Summary
    total = len(results)
    solved = sum(1 for r in results if any(sample['success'] for sample in r['samples']))
    print(f"\n✅ Solved {solved}/{total} problems.")
    print(f"🔧 Using ports: {port_list}")
    print(f"🔧 Using model: {MODEL_NAME} Dataset: {args.dataset}")

    sanitized_model_name = helpers.sanitize_filename(MODEL_NAME.split("/")[-1])
#    filename = f"{sanitized_model_name}_results_atk{args.at_k}"
    filename = f"{sanitized_model_name}_{dataset_name}_{args.split}_{args.lang}_atk{args.at_k}_sidx_{args.start_idx}_n{args.max_problems}_results"

    if args.with_reasoning:
        filename += "_with_reasoning"
    filename_summary = filename + "_summary.json"
    filename += ".json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    pass_at_k = defaultdict(int)
    for r in results:
        success_at = next((s["sample_id"] for s in r["samples"] if s["success"]), None)
        if success_at:
            for k in range(success_at, args.at_k + 1):
                pass_at_k[k] += 1

    print("\n📊 Cumulative Pass@k Results:")
    for k in range(1, args.at_k + 1):
        count = pass_at_k.get(k, 0)
        print(f"  ✅ Pass@{k}: {count}/{len(results)} ({count / len(results) * 100:.1f}%)")

    with open(filename_summary, "w") as f:
        json.dump({
            "pass@k": {str(k): pass_at_k.get(k, 0) for k in range(1, args.at_k + 1)},
            "total": len(results)
        }, f, indent=2)

    print(f"\n📝 Summary written to {filename_summary}")
if __name__ == "__main__":
    main()