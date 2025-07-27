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




def load_problems():
    dataset = load_dataset("Qwen/CodeElo", split="test")
    return list(dataset)  # Convert to list of dicts



def query_qwen_model(prompt: str, k: int,temperature:float=1.0, endpoint: str = None) -> list[str]:
    msg_short="You are a helpful coding assistant."
    msg_long="""You are a helpful coding assistant. Your role as an assistant involves
 thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions.
 This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing,
 and iteration to develop well-considered thinking process. Please structure your response into two main sections: Reasoning and
 Solution using the specified format: <think> Reasoning section </think> Coding section. In the Reasoning section, detail your
 reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant
 findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps.
 In the Coding section, based on various attempts, explorations, and reflections from the Reasoning section, systematically
 present the final solution that you deem correct. The Coding section should contain code only. Now, try to solve the following question through the above guidelines"""
    print('query is called with',k,'model_name',MODEL_NAME,'temp',temperature)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": msg_short},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "repetition_penalty": 1.1,
        "max_tokens": 8092,
        "do_sample":True,
        "top_p":0.95,
        "n": k  # <-- IMPORTANT: request k completions
    }
#    print('len if payload',len(payload))
    response = requests.post(endpoint, json=payload)

#    response = requests.post(VLLM_ENDPOINT, json=payload)
    response.raise_for_status()
    print('returned',response.json()["choices"])
    result = response.json()

    # Always return a list of `k` code completions
    completions = [choice["message"]["content"] for choice in result["choices"]]
#    print('completions are',len(completions),completions)
    return completions
#    return [choice["message"]["content"] for choice in response.json()["choices"]]


   


def main():
    print('hello!')
    problems = load_problems()
    parser = argparse.ArgumentParser()
    parser.add_argument('--at_k', type=int, default=1, help='Number of completions to use for pass@k evaluation')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--lang', type=str, default='cpp')
    #parser.add_argument('--ports', type=lambda s: list(map(int, s.split(','))), default=[8000])
    parser.add_argument('--port', type=str, default="8000", help='Comma-separated list of ports for vLLM servers')
    parser.add_argument('--max_problems', type=int, default=10)
    parser.add_argument('--with_reasoning', dest='with_reasoning', action='store_true', help='Include reasoning in the prompt')
    parser.add_argument('--no_reasoning', dest='with_reasoning', action='store_false', help='Exclude reasoning in the prompt')
    parser.set_defaults(with_reasoning=True)
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature for model generation')

    args = parser.parse_args()
    port_list = [int(p.strip()) for p in args.port.split(",")]
    print(f"🔧 Using ports: {port_list}")
    print(f"🔧 Language: {args.lang}")
    problems = problems[:args.max_problems]

    global MODEL_NAME
    MODEL_NAME = args.model_name or get_model_name_from_vllm(port_list[0])
    print(f"🔧 Using model: {MODEL_NAME}")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import itertools

    port_cycle = itertools.cycle(port_list)
    def make_model(port):
        endpoint = f"http://localhost:{port}/v1/chat/completions"
        return lambda prompt, k=args.at_k, temperature=args.temperature: query_qwen_model(
            prompt, k=k, temperature=temperature, endpoint=endpoint
        )

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

    sanitized_model_name = helpers.sanitize_filename(MODEL_NAME.split("/")[-1])
#    filename = f"{sanitized_model_name}_results_atk{args.at_k}"
    filename = f"{sanitized_model_name}_{args.lang}_atk{args.at_k}_n{args.max_problems}_results"

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