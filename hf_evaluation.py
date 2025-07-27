import os
import argparse
import json
from collections import defaultdict
from datasets import load_dataset
from helpers import (
    generate_prompt_from_problems,
    get_llm_output_on_text_batch,
    extract_code,
    run_code_locally,
    normalize_output,
    sanitize_filename,
    compile_code,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def evaluate_model_on_dataset(model, tokenizer, dataset, model_name, device, *,
                               lang='cpp',
                               batch_size=8,
                               temp=0.001,
                               with_reasoning=True,
                               append_to_answer='',
                               max_new_tokens=8192,
                               filename_prefix=None,
                               at_k=1,
                               instruct_flag=True,
                               verbose=False):
    prompts = generate_prompt_from_problems(dataset[:], lang=lang, with_reasoning=with_reasoning)
    #print('number of prompts',len(prompts),type(prompts))
    config = {
        "temperature": temp,
        "do_sample": True,
        "max_new_tokens": max_new_tokens
    }
    if 'problem_id' in dataset[0].keys():
        problem_key='problem_id'
    else:
        problem_key='id'

    completions, _ = get_llm_output_on_text_batch(
        model,
        prompts,
        tokenizer,
        device,
        append_to_answer=append_to_answer,
        micro_batch_size=batch_size,
        mode='default',
        incontext_mode=-1,
        moveflag=True,
        Tree=None,
        print_rel=False,
        verbose=verbose,
        instruct_flag=instruct_flag,
        at_k=at_k,
        **config
    )
    num_problems=len(completions)//at_k
   # print(len(completions))
    print("Generated", len(prompts), "prompts")
    print("Received", len(completions), "outputs")
    print('number of problems',num_problems)
    all_results = []#[[]] *num_problems 
    current_pid=None
    samples=[]
    solved=False
    for i, code in enumerate(completions):
        problem_idx=i//at_k
        problem=dataset[problem_idx]
        print('problem idx',problem_idx,'current_pid',current_pid)
        print('has problmem',problem_idx,'been solved?',solved)
        if current_pid is None:
            current_pid = problem_idx
        if problem_idx != current_pid:
            print('starting a new problem')
            # store finished problem block
            all_results.append(
                {"problem_id": current_pid, "samples": samples}
            )
            samples = []
            solved=False
            current_pid = problem_idx    
        elif solved:
            continue
     #   print('get test cases for problem')
        tests = problem.get('official_tests') or problem.get('examples', [])
 #       print(f"\n--- Code Generated (sample {i+1}) ---\n{code}\n")
        if isinstance(code, list):  # defensive programming
  #          print('len(code)',len(code))
  #          print('list detected',len(code))
            code = code[0]
        extracted = extract_code(code,lang=lang)
#        print(f"---- Code Extracted ----\n{extracted}\n")

        passed = 0
        compiled_flag=False
        try:
            tmpdir, code_path,err = compile_code(extracted, lang)
            compiled_flag=True
        except Exception as e:
#              print(f"⚠️ Error running code: {e}")
            print(f"code does not compile")
            actual = "ERROR"                
        total = len(tests)
        if compiled_flag:            
            for test in tests:
                if isinstance(test, dict):
                    input_str = test.get('input', '')
                    expected_output = test.get('output', '')
                else:
                    input_str, expected_output = test
                try:
#                    tmpdir, code_path = compile_code(extracted, lang)
                    actual_output = run_code_locally(code_path, input_str,lang=lang)
                    actual = normalize_output(actual_output)
                except Exception as e:
    #              print(f"⚠️ Error running code: {e}")
                    actual = "ERROR"                
                expected = normalize_output(expected_output)

#                print('actual output:', actual)
 #               print('expected output:', expected)
                if actual == expected:
                    passed += 1
                else:
                    break
        samples.append({
            "problem_id": problem[problem_key],
            "sample_id": i%at_k + 1,
            "passed": passed,
            "total": total,
            "success": passed == total
        })

        if passed == total:
            solved=True
            print(f"✅ Solved {problem[problem_key]} with sample {i%at_k+1}/{at_k}")
           # break

    # push the last problem
    if samples:
        all_results.append(
            {"problem_id": current_pid, "samples": samples}
        )
                
    return all_results 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or path (e.g. Qwen/Qwen2.5-Coder-7B-Instruct)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., Qwen/CodeElo or open-r1/codeforces)")
    parser.add_argument("--split", type=str, default="test", help="Split to use (e.g., train/test)")
    parser.add_argument("--max_problems", type=int, default=10, help="Number of problems to evaluate")
    parser.add_argument("--lang", type=str, default="cpp")
    parser.add_argument("--at_k", type=int, default=1)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument("--instruct_flag", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--with_reasoning", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print('instruct flag is on',args.instruct_flag)
    print('at_k',args.at_k)
    ACCESS_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=access_token,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=access_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    model = torch.compile(model)
    dataset_name=args.dataset.split('/')[-1]

#    dataset = load_dataset(args.dataset, split=args.dataset_split)[:args.max_problems]
    dataset = load_dataset(args.dataset, split=args.split).select(range(args.max_problems))

    results=evaluate_model_on_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        model_name=args.model_name,
        device=args.device,
        lang=args.lang,
        batch_size=args.batch_size,
        temp=args.temperature,
        at_k=args.at_k,
        instruct_flag=args.instruct_flag,
        with_reasoning=args.with_reasoning,
        verbose=args.verbose
    )
    total = len(results)

    sanitized_model_name = sanitize_filename(args.model_name.split("/")[-1])
#    filename = filename_prefix or f"{sanitized_model_name}_{lang}_atk{at_k}_n{total}_results"
    filename = f"hf_{sanitized_model_name}_{dataset_name}_{args.split}_{args.lang}_atk{args.at_k}_sidx_{args.start_idx}_n{args.max_problems}_results"
    if args.with_reasoning:
        filename += "_with_reasoning"

    filename_json = filename + ".json"
    filename_summary = filename + "_summary.json"

    os.makedirs("results", exist_ok=True)
    with open(f"results/{filename_json}", "w") as f:
        json.dump(results, f, indent=2)

    # Pass@k stats
    pass_at_k = defaultdict(int)
    if isinstance(results, dict):
        results = [results]          # wrap single dict into a list
    for r in results:
        success_at = next((s["sample_id"] for s in r["samples"] if s["success"]), None)
        if success_at:
            for k in range(success_at, args.at_k + 1):
                pass_at_k[k] += 1

    print("\n📊 Cumulative Pass@k Results:")
    for k in range(1, args.at_k + 1):
        count = pass_at_k.get(k, 0)
        print(f"  ✅ Pass@{k}: {count}/{total} ({count / total * 100:.1f}%)")

    with open(f"results/{filename_summary}", "w") as f:
        json.dump({
            "pass@k": {str(k): pass_at_k.get(k, 0) for k in range(1, args.at_k + 1)},
            "total": total
        }, f, indent=2)

    print(f"\n📝 Summary written to results/{filename_summary}")
