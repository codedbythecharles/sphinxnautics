import os
import torch
import re 
import random
import subprocess
import tempfile
import inspect
from accelerate import Accelerator
from transformers.modeling_outputs import CausalLMOutput
import re
from typing import Optional
import time
import shutil
import subprocess
from ddp_debugger import ddp_debug
import torch.nn.functional as F


incontext_mode_to_key={0:'zero',1:'short',2:'long',-1:'m1',-2:'m2'}

 
def sanitize_filename(name):
    return re.sub(r'[^\w\-_.]', '_', name)


# Accept common aliases for C++ / Python
LANG_ALIASES = {
    "cpp": {"cpp", "c++", "cc", "cxx"},
    "python": {"python", "py"},
}

# Full fenced code blocks:  ```lang\n ... \n```
CLOSED_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+.\-]*)\s*(.*?)```", re.S)

# Start fence when closing is missing:  ```lang\n ... (until EOF)
OPEN_FENCE_START_RE = re.compile(r"```([a-zA-Z0-9_+.\-]*)\s*(.*)", re.S)
def _lang_matches(found_lang: str, target_lang: str) -> bool:
    found = found_lang.lower().strip()
    target = target_lang.lower().strip()
    # Exact match or alias match
    return (found == target) or (target in LANG_ALIASES and found in LANG_ALIASES[target])



def extract_code(raw_output: str, lang: str = "cpp") -> Optional[str]:
    """Extract the code in the requested language from a response string.
    """
    # 1) Search for properly *closed* fenced code blocks
    if not raw_output:
        return ''
    matches = CLOSED_FENCE_RE.findall(raw_output)
    if matches:
        # Prefer target language
        for fence_lang, code in matches:
            if _lang_matches(fence_lang, lang):
                return code.strip()

        # If target language wasn't found, fall back to the first code block
        first_code = matches[0][1].strip()
        if first_code:
            return first_code

    # 2) If nothing, try "open fence" without closing ```
    open_match = OPEN_FENCE_START_RE.search(raw_output)
    if open_match:
        fence_lang, code_so_far = open_match.groups()
        if _lang_matches(fence_lang, lang):
            return code_so_far.strip()

    # 3) Nothing found
    return None

def extract_code2(raw_output, lang='cpp',mode='codeforces'):
    if not raw_output:
        return ''
    if mode=='codeforces':
        extract_code(raw_output,lang)
        
    else:
        if lang == 'python':
            # Match from `def solve():` to the end of script (including optional main guard)
            match = re.search(r"(def solve\(\):.*?if __name__ == .__main__.:.*?solve\(\))", raw_output, re.DOTALL)
            if not match:
                # Try fallback: just match solve + call
                match = re.search(r"(def solve\(\):.*?solve\(\))", raw_output, re.DOTALL)
        elif lang == 'cpp':
            # Match a full C++ program from #include or main
            match = re.search(r"(#include\s+<.*?>.*?int main\(\).*?\{.*?\n\})", raw_output, re.DOTALL)
            if not match:
                match = re.search(r"(int main\(\).*?\{.*?\n\})", raw_output, re.DOTALL)
        else:
            match = None

        if match:
            return match.group(1).strip()

        # Fallback: return raw output as-is
        return raw_output.strip()

from typing import Tuple, Optional

def collate_fn(batch):
#    for b in batch:
 #       print('.........',len(batch),b.keys(),b['text'],b['output_texts'])
  #  [torch.tensor(example['input_ids']) for example in batch]
    input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
    labels = torch.stack([torch.tensor(example['labels']) for example in batch])
    ideal_outputs=torch.stack([torch.tensor(example['ideal_outputs']) for example in batch])
    attention_mask=torch.stack([torch.tensor(example['attention_mask']) for example in batch])
#    encoded_prompt=torch.stack([torch.tensor(example['encoded_prompt']['input_ids']) for example in batch])
#    print('reached!')
    return {'input_ids': input_ids, 'labels': labels,'ideal_outputs': ideal_outputs,'attention_mask':attention_mask}#,'encoded_prompt':encoded_prompt}

def collate_fn_pad(batch, pad_token_id=0):
    # Convert lists of examples to tensors
    input_ids = [torch.tensor(example['input_ids']).squeeze(0) if len(torch.tensor(example['input_ids']).size()) > 1 else torch.tensor(example['input_ids']) for example in batch]
    labels = [torch.tensor(example['labels']).squeeze(0) if len(torch.tensor(example['labels']).size()) > 1 else torch.tensor(example['labels']) for example in batch]
#    ideal_outputs = [torch.tensor(example['ideal_outputs']).squeeze(0) if len(torch.tensor(example['ideal_outputs']).size()) > 1 else torch.tensor(example['ideal_outputs']) for example in batch]
    attention_mask = [torch.tensor(example['attention_mask']).squeeze(0) if len(torch.tensor(example['attention_mask']).size()) > 1 else torch.tensor(example['attention_mask']) for example in batch]

    # Pad sequences to the same length
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignored labels
  #  ideal_outputs_padded = torch.nn.utils.rnn.pad_sequence(ideal_outputs, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Return the padded batch
    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded,
#        'ideal_outputs': ideal_outputs_padded,
        'attention_mask': attention_mask_padded
    }

def compile_code(code: str, lang: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Compiles or prepares the code file depending on language.
    """
    tmpdir = tempfile.mkdtemp()

    if lang == "cpp":
        code_path = os.path.join(tmpdir, "solution.cpp")
        exec_path = os.path.join(tmpdir, "solution")
        with open(code_path, "w") as f:
            f.write(code)

        compile_proc = subprocess.run(
            ["g++", "-O2", "-std=c++17", code_path, "-o", exec_path],
            capture_output=True, text=True
        )
        if compile_proc.returncode != 0:
            return None, None, f"[Compilation failed]\n{compile_proc.stderr.strip()}"
        return tmpdir, exec_path, None

    elif lang == "python":
        code_path = os.path.join(tmpdir, "solution.py")
        with open(code_path, "w") as f:
            f.write(code)
        return tmpdir, code_path, None

    else:
        return None, None, "[Unsupported language]"


def run_code_locally(exec_path: str, input_str: str, lang="python", timeout=8) -> str:
    try:
        if lang == "cpp":
            run_proc = subprocess.run(
                [exec_path],
                input=input_str,
                text=True,
                capture_output=True,
                timeout=timeout
            )
        elif lang == "python":
            run_proc = subprocess.run(
                ["python3", exec_path],
                input=input_str,
                text=True,
                capture_output=True,
                timeout=timeout
            )
        else:
            return "[Unsupported language]"

        if run_proc.returncode == 0:
            return run_proc.stdout.strip()
        else:
            return f"[Runtime error]\n{run_proc.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return "[Execution timed out]"

def normalize_output(output: str):
    lines = output.strip().splitlines()
    return "\n".join(line.strip() for line in lines)

coding_prompts={'python':'You will be given a competitive programming problem.\nAnalyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution. Please reason step by step about your solution approach, then provide a complete implementation in Python 3 that is thoroughly optimized for both speed and memory usage.\n\nYour solution must read input from standard input (input()), write output to standard output (print()).\nDo not include any debug prints or additional output.\n\nPut your final solution within a single code block:\n```python\n<your code here>\n```',
  'cpp':'You will be given a competitive programming problem.\nAnalyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution. Please reason step by step about your solution approach, then provide a complete implementation in C++17 that is thoroughly optimized for both speed and memory usage.\n\nYour solution must read input from standard input (cin), write output to standard output (cout).\nDo not include any debug prints or additional output.\n\nPut your final solution within a single code block:\n```cpp\n<your code here>\n```',}

#alternative shorter version
coding_prompts_shorter={'python':'You are a helpful coding assistant with advanced reasonign and algorithmic skills. You write your codes in Python 3. Your solution must read input from standard input (input()), write output to standard output (print()).\nDo not include any debug prints or additional output.\n\nPut your final solution within a single code block:\n```python\n<your code here>\n```',
  'cpp':'You are a helpful coding assistant with advanced reasonign and algorithmic skills. You write codes in C++17.\n\nYour solution must read input from standard input (cin), write output to standard output (cout).\nDo not include any debug prints or additional output.\n\nPut your final solution within a single code block:\n```cpp\n<your code here>\n```',}

def configure_optimizers(m, weight_decay,print_it=True):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in m.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. 
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if print_it:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        return optim_groups


# Initialize accelerator
accelerator = Accelerator()

# Move model and dataloader to the proper device using the accelerator
def evaluate_loss(model,dataloader,pad_token_id,device,num_test_epochs=1,moveflag=True,print_every=100):
    model, dataloader = accelerator.prepare(model, dataloader)
    pad_token_id=pad_token_id
    total_test_steps=0
    epoch_loss=0
    model.eval()
#    model = model.to(device)
    for idx, batch in enumerate(dataloader):
        total_test_steps += 1        
        inputs = batch['input_ids'][:,:]
        attention_mask = batch['attention_mask'][:,:]
        labels = batch['labels'][:,:]
        B=inputs.shape[0]
#        position_ids = torch.arange(0, inputs.size(1), dtype=torch.long, device=device).unsqueeze(0).expand_as(inputs)
        if moveflag:
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels=labels.to(device)
 #           position_ids=position_ids.to(device)
        with torch.no_grad():            
            outputs = model(input_ids=inputs, labels=labels,attention_mask=attention_mask)#,position_ids=position_ids)
#            print('shape',outputs.loss.item())
            epoch_loss+=outputs.loss.mean().item()
        if idx%print_every==0:
            print('avg validation loss:',epoch_loss/total_test_steps)
#        print(inputs.shape)
    model.train()
    return epoch_loss/total_test_steps


#distillation/RL with a teacher
def distill_simerr(model,teacher,dataloader,optimizer,scheduler,device,pad_token_id,eos_token_id,num_helper_samples=5,optimizer_helper=None,helper=None,move_to_device=False,max_num_steps=None,dataloader_test=None,horizon=1,gradient_accumulation_steps=8,num_epochs=1,num_test_batches=30,print_every=10,batch_size=8,lah=0,temp=0.001,max_new_tokens=256,tokenizer=None,do_sample=False,with_ddp=False,ddp_rank=0,ddp_world_size=1,device_type='cuda',checkpoint_every=500,filename='model',writer=None,past_epoch_steps=0,checkpoint_dir=None,start_step=0,unfreeze_idx=0,activate_rl=True,reverse_kl=False,kl_temp=1,VERBOSE=False):
    losses=[]
    total_steps=start_step
    total_processes_samples=0
    batch_size=dataloader.batch_size
    max_num_steps=max_num_steps//batch_size
    model.train()
    num_processed_tokens=[]
    for epoch in range(num_epochs):  # 3 epochs
        epoch_steps=0
        model.train()
        epoch_loss = 0
        device=model.device
        loss_local = torch.tensor(0.0, device=device)
        optimizer.zero_grad(set_to_none=True)        
#        optimizer.zero_grad()  # Zero out gradients before accumulation
        if optimizer_helper is not None:
            optimizer_helper.zero_grad(set_to_none=True)

        t0=time.time()
        if ddp_rank==0:
            print('-------------',len(dataloader))
        for idx, batch in enumerate(dataloader):
            epoch_steps+=1
            B,T=batch['input_ids'].shape
            num_processed_tokens.append((B*T)*ddp_world_size)
            total_steps += 1
            
            if move_to_device:
                inputs = batch['input_ids'][:,:].to(device)
                attention_mask = batch['attention_mask'][:,:].to(device)
                labels = batch['labels'][:,:].to(device)
            else:
                inputs = batch['input_ids'][:,:]
                attention_mask = batch['attention_mask'][:,:]
                labels = batch['labels'][:,:]
            if idx % print_every == 0 and ddp_rank==0:
                if idx == 0:
                    print('--- input shape?', idx, batch['input_ids'].shape)
                if idx > 0:
                    print('id:', idx, batch['input_ids'].shape,'total processes samples',total_processes_samples,'avg loss over the last 10',sum(losses[-10:]) / len(losses[-10:]))
            total_processes_samples+=batch_size

            current_length=inputs.shape[-1]#$np.min([outputs.shape[-1],inputs.shape[-1]])

            inputs = inputs[:,:current_length]#outputs
            attention_mask=attention_mask[:,:current_length]
            labels=labels[:,:current_length]
            
            # Assuming `outputs` is the model output
            if lah>0:
                last_index=torch.sum(attention_mask,dim=-1)-1
                B=attention_mask.shape[0]
                seq_len = labels.size(1)  # Maximum sequence length
                range_tensor = torch.arange(seq_len, device=labels.device).unsqueeze(0)  # Shape: (1, seq_len)

                # Broadcast and compare against `last_index`
                # Mask tokens that are beyond the last valid index or are -100 in the labels
                answer_mask = (range_tensor <= last_index.unsqueeze(1)) & (labels != -100)
                question_mask = (range_tensor <= last_index.unsqueeze(1)) & (labels == -100)
                last_index=torch.sum(answer_mask,dim=-1)-1
                last_question_index=torch.sum(question_mask,dim=-1)
                answer_ids = torch.masked_select(inputs, answer_mask)
                num_el=256#answer_ids.shape[0]#labels[0:,last_question_index:].shape[-1]
                min_len=10#inputs.shape[-1]-5#int(inputs.shape[-1]-last_question_index)
          #      max_len=inputs.shape[-1]
                if activate_rl:    
                    with torch.no_grad(),torch.autocast(device_type="cuda", dtype=torch.bfloat16):  
                        s_device=model.device     
                        outputs = model.generate(input_ids=inputs[0:,:last_question_index].to(s_device),attention_mask=attention_mask[0:,:last_question_index].to(s_device), temperature=temp,pad_token_id=pad_token_id, top_p=0.9,top_k=50,min_length=min_len, max_new_tokens=max_new_tokens,return_dict_in_generate=True,output_scores=True,do_sample=do_sample)
#                        s_logits = outputs.logits[:, -1, :]
                        new_inputs=outputs['sequences']#torch.cat([inputs[0:,:last_question_index],]

                    # compute generator and verifier logits: 
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):       
                        attention_mask = torch.ones_like(new_inputs, dtype=torch.long)
                        t_device=teacher.device
                        t_outputs = teacher(input_ids=new_inputs.to(t_device),attention_mask=attention_mask)
                        t_logits=t_outputs.logits[:,last_question_index:]
                        t_logp = F.log_softmax(t_logits / kl_temp, dim=-1)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):       
                        s_device=model.device
                        s_outputs = model(input_ids=new_inputs.to(s_device),attention_mask=attention_mask)
                        s_logits=s_outputs.logits[:,last_question_index:]
                        s_logp = F.log_softmax(s_logits / kl_temp, dim=-1)
                        if reverse_kl:
                            loss_kl = F.kl_div(s_logp, t_logp.to(s_device), reduction="batchmean", log_target=True) * (kl_temp **2)
                        else:
                            loss_kl = F.kl_div(t_logp.to(s_device), s_logp, reduction="batchmean", log_target=True) * (kl_temp **2)                        
                        loss = loss_kl / gradient_accumulation_steps
                else:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        t_device=teacher.device                        
                        outputs = teacher.generate(input_ids=inputs[0:,:last_question_index].to(t_device),attention_mask=attention_mask[0:,:last_question_index].to(t_device), temperature=temp,pad_token_id=pad_token_id, top_p=0.9,top_k=50,min_length=min_len, max_new_tokens=max_new_tokens,return_dict_in_generate=True,output_scores=True,do_sample=do_sample)
                        new_inputs=outputs['sequences']#torch.cat([inputs[0:,:last_question_index],]
                        t_logits=torch.stack(outputs['scores'],dim=1).to(device) 
                        t_logp = F.log_softmax(t_logits / kl_temp, dim=-1)
                        
                    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                         attention_mask = torch.ones_like(new_inputs, dtype=torch.long)
                         s_device=model.device
                         old_inputs=inputs[0:,:last_question_index]
                         all_inputs=torch.cat([old_inputs.to(s_device),new_inputs.to(s_device)],dim=-1)
                         all_labels=all_inputs.clone()
                         all_labels[:,:last_question_index]=torch.tensor(-100,device=s_device)                         
                        # import code;code.interact(local=locals())
                         outputs = model(input_ids=all_inputs, labels=all_labels,attention_mask=attention_mask)
                         loss_CE=outputs.loss
 #                       if reverse_kl:
#                            loss_kl = F.kl_div(s_logp, t_logp.to(s_device), reduction="batchmean", log_target=True) * (kl_temp **2)
  #                      else:
   #                         loss_kl = F.kl_div(t_logp.to(s_device), s_logp, reduction="batchmean", log_target=True) * (kl_temp **2)
#                loss_kl = F.kl_div(s_logp, t_logp, reduction="batchmean", log_target=True) * (kl_temp **2)
                         loss = loss_CE / gradient_accumulation_steps
#                        import code;code.interact(local=locals())
            else:
                with torch.autocast(device_type=device,dtype=torch.bfloat16):
                    outputs = model(input_ids=inputs, labels=labels,attention_mask=attention_mask,use_cache=False)
                    loss = outputs.loss / gradient_accumulation_steps

            loss.backward()
            loss_local+=loss.detach()
            head_mean= model.lm_head.weight.mean().detach()
            head_norm=torch.norm(model.lm_head.weight).detach()
            grad0=model.lm_head.weight.grad
            grad_norm=torch.norm(grad0).detach()
            if VERBOSE and total_steps%print_every==0:
                print("model.module.lm_head.weight.requires_grad",model.lm_head.weight.requires_grad)
                print("Gradient norm head:", grad0.norm().item() if grad0 is not None else "None")
                print('Norm of head',head_norm)
                print('Mean of head',head_mean)
                print('First element head',model.lm_head.weight[0][0])
            if (idx + 1) % gradient_accumulation_steps == 0:
                print('optimizer taking a step!')
                optimizer.step()
                current_lr = optimizer.param_groups[0]['lr']
                if ddp_rank==0:
                    print("Current LR:", current_lr)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                sum_processed_tokens=sum(num_processed_tokens[-gradient_accumulation_steps:])
                if device_type=='cuda':
                    torch.cuda.synchronize()
                t1=time.time()
                dt=t1-t0
                print('tokens/sec',sum_processed_tokens/dt*ddp_world_size)
                t0=time.time()
                if writer is not None:
                    writer.add_scalar("Loss/train", loss_local, past_epoch_steps+total_steps)
                    writer.add_scalar("GradNorm/lm_head", grad_norm, past_epoch_steps+total_steps)
                    writer.add_scalar("HeadMean/lm_head", head_mean, past_epoch_steps+total_steps)
                    writer.add_scalar("HeadNorm/lm_head", head_norm, past_epoch_steps+total_steps)
                    writer.add_scalar("LR", current_lr, past_epoch_steps+total_steps)
                    # Save model checkpoint
                torch.cuda.empty_cache()    
                loss_local=0
            if ddp_rank==0 and checkpoint_dir is not None:
                if total_steps % checkpoint_every == 0:
                    torch.save({
                        "step": total_steps,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()
                    }, os.path.join(checkpoint_dir, f"{filename}_step_{total_steps}.pt"))                    
            """"
            logits = outputs.logits

            # Compute the loss manually with reduction='none'
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss_per_element = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Reshape loss_per_element back to the batch size
            loss_per_element = loss_per_element.view(labels.size(0), -1).mean(dim=1)

            # You can now use this to scale or accumulate gradients
            scaled_loss = loss_per_element / gradient_accumulation_steps
            # To get the overall loss as before (if needed)
            #loss = loss_per_element.mean() / gradient_accumulation_steps
#            loss = outputs.loss / gradient_accumulation_steps  # Scale the loss by accumulation steps
#            print(scaled_loss)
            batch_size=B
            for b in range(batch_size):
                scaled_loss[b].backward(retain_graph=True if b < batch_size - 1 else False)
#                retain_graph=True if b < batch_size - 1 else False
            """

            epoch_loss += loss.mean().item() * gradient_accumulation_steps  # Multiply back to the original scale
            losses.append(loss.mean().item() * gradient_accumulation_steps)
            if (max_num_steps is not None) and (total_steps>=max_num_steps):
                break

        if dataloader_test is not None:
            loss_eval=evaluate_loss(model,dataloader_test,pad_token_id,device,num_test_epochs=1)

        avg_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch + 1}/{num_epochs}, train Loss: {avg_loss})")#, test loss:{loss_eval}")
    if ddp_rank==0 and checkpoint_dir is not None:
        print('writing to disk with',checkpoint_dir)
        torch.save({
            "step": total_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }, os.path.join(checkpoint_dir, f"{filename}_step_{total_steps}.pt"))     
    return total_steps



def pretrain_simerr(model,dataloader,optimizer,scheduler,device,pad_token_id,eos_token_id,num_helper_samples=5,optimizer_helper=None,helper=None,move_to_device=False,max_num_steps=None,dataloader_test=None,horizon=1,gradient_accumulation_steps=8,num_epochs=1,num_test_batches=30,print_per_batch=10,batch_size=8,lah=0,temp=0.001,max_new_tokens=256,tokenizer=None,do_sample=False,with_ddp=True,ddp_rank=0,ddp_world_size=1,device_type='cuda',checkpoint_every=500,filename='model',writer=None,past_epoch_steps=0,checkpoint_dir=None,start_step=0,unfreeze_idx=0,eval_every=100,verbose=False):
    losses=[]
    total_steps=start_step
    total_processes_samples=0
    batch_size=dataloader.batch_size
    max_num_steps=max_num_steps//batch_size
    model.train()
    num_processed_tokens=[]
    for epoch in range(num_epochs):  # 3 epochs
        epoch_steps=0
        model.train()
        epoch_loss = 0
        loss_local = torch.tensor(0.0, device=device)
        optimizer.zero_grad(set_to_none=True)        
#        optimizer.zero_grad()  # Zero out gradients before accumulation
        if optimizer_helper is not None:
            optimizer_helper.zero_grad(set_to_none=True)

        t0=time.time()
        if ddp_rank==0:
            print('-------------',len(dataloader))
        for idx, batch in enumerate(dataloader):
            epoch_steps+=1
            B,T=batch['input_ids'].shape
            num_processed_tokens.append((B*T)*ddp_world_size)
            total_steps += 1
            
            if move_to_device:
                inputs = batch['input_ids'][:,:].to(device)
                attention_mask = batch['attention_mask'][:,:].to(device)
                labels = batch['labels'][:,:].to(device)
            else:
                inputs = batch['input_ids'][:,:]
                attention_mask = batch['attention_mask'][:,:]
                labels = batch['labels'][:,:]
            if idx % print_per_batch == 0 and ddp_rank==0:
                if idx == 0:
                    print('---first sample. input shape is', batch['input_ids'].shape)
                if idx > 0:
                    print('id:', idx, batch['input_ids'].shape, sum(losses[-print_per_batch:]) / print_per_batch,'total processes samples',total_processes_samples)
            total_processes_samples+=batch_size

            current_length=inputs.shape[-1]#$np.min([outputs.shape[-1],inputs.shape[-1]])

            inputs = inputs[:,:current_length]#outputs
            attention_mask=attention_mask[:,:current_length]
            labels=labels[:,:current_length]
            
            with torch.autocast(device_type=device,dtype=torch.bfloat16):
                outputs = model(input_ids=inputs, labels=labels,attention_mask=attention_mask,use_cache=False)
                loss = outputs.loss / gradient_accumulation_steps
                
            if with_ddp:
                model.require_backward_grad_sync = ((idx + 1) % gradient_accumulation_steps == 0)

            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_accumulation_steps)            
            loss_local+=loss.detach()
  #          grad = model.module.model.layers[-1].self_attn.q_proj.weight.grad
            if ddp_rank==0 and unfreeze_idx>0:
                grad_last = model.module.model.layers[-unfreeze_idx].self_attn.q_proj.weight.grad
            grad0 = model.module.lm_head.weight.grad
#            print("Grad mean:", grad0.mean().item() if grad0 is not None else "None")
            head_mean= model.module.lm_head.weight.mean().detach()
            head_norm=torch.norm(model.module.lm_head.weight).detach()
            grad_norm=grad0.norm().detach()
   #         print("Gradient norm:", grad.norm().item() if grad is not None else "None")
            if ddp_rank==0 and verbose and total_steps%print_per_batch==0:
                print("model.module.lm_head.weight.requires_grad",model.module.lm_head.weight.requires_grad)
                print("Gradient norm head:", grad0.norm().item() if grad0 is not None else "None")
    #            print("Gradient norm2:", grad2.norm().item() if grad is not None else "None")
            # print('norm',norm)
                print(f"Gradient norm qproj layer -{unfreeze_idx}:", grad_last.norm().item() if grad_last is not None else "None")
                print('Norm of head',head_norm)
                print('Mean of head',head_mean)
                print('First element head',model.module.lm_head.weight[0][0])
#            import code;code.interact(local=locals())
#            if ddp:
#    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            if (idx + 1) % gradient_accumulation_steps == 0:
                
                optimizer.step()
                current_lr = optimizer.param_groups[0]['lr']
                if ddp_rank==0:
                    print("Current LR:", current_lr)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                sum_processed_tokens=sum(num_processed_tokens[-gradient_accumulation_steps:])
                torch.distributed.all_reduce(loss_local, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(head_mean, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(grad_norm, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(head_norm, op=torch.distributed.ReduceOp.AVG)
                if ddp_rank==0:
                    if device_type=='cuda':
                        torch.cuda.synchronize()
                    t1=time.time()
                    dt=t1-t0
                    print('tokens/sec',sum_processed_tokens/dt*ddp_world_size)
                    t0=time.time()
                    if writer is not None:
                        writer.add_scalar("Loss/train", loss_local, past_epoch_steps+total_steps)
                        writer.add_scalar("GradNorm/lm_head", grad_norm, past_epoch_steps+total_steps)
                        writer.add_scalar("HeadMean/lm_head", head_mean, past_epoch_steps+total_steps)
                        writer.add_scalar("HeadNorm/lm_head", head_norm, past_epoch_steps+total_steps)
                        writer.add_scalar("LR", current_lr, past_epoch_steps+total_steps)
                    # Save model checkpoint
                torch.cuda.empty_cache()    
                loss_local=0
            if ddp_rank==0 and checkpoint_dir is not None:
                if total_steps % checkpoint_every == 0:
                    torch.save({
                        "step": total_steps,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()
                    }, os.path.join(checkpoint_dir, f"{filename}_step_{total_steps}.pt"))                    
            """"
            logits = outputs.logits

            # Compute the loss manually with reduction='none'
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss_per_element = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Reshape loss_per_element back to the batch size
            loss_per_element = loss_per_element.view(labels.size(0), -1).mean(dim=1)

            # You can now use this to scale or accumulate gradients
            scaled_loss = loss_per_element / gradient_accumulation_steps
            # To get the overall loss as before (if needed)
            #loss = loss_per_element.mean() / gradient_accumulation_steps
#            loss = outputs.loss / gradient_accumulation_steps  # Scale the loss by accumulation steps
#            print(scaled_loss)
            batch_size=B
            for b in range(batch_size):
                scaled_loss[b].backward(retain_graph=True if b < batch_size - 1 else False)
#                retain_graph=True if b < batch_size - 1 else False
            """

            epoch_loss += loss.mean().item() * gradient_accumulation_steps  # Multiply back to the original scale
            losses.append(loss.mean().item() * gradient_accumulation_steps)
            if (max_num_steps is not None) and (total_steps>=max_num_steps):
                break
            if dataloader_test is not None and total_steps%eval_every==0:
                loss_eval=torch.tensor(evaluate_loss(model,dataloader_test,pad_token_id,device,num_test_epochs=1),device=device)
                torch.distributed.all_reduce(loss_eval, op=torch.distributed.ReduceOp.AVG)
                if ddp_rank==0:
                    print(f"Step {epoch + 1}/{num_epochs}, val Loss: {loss_eval})")#, test loss:{loss_eval}")
                    if writer is not None:
                        writer.add_scalar("Loss/val", loss_eval, past_epoch_steps+total_steps)
                                    
        avg_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch + 1}/{num_epochs}, train Loss: {avg_loss})")#, test loss:{loss_eval}")
    if ddp_rank==0 and checkpoint_dir is not None:
        print('writing to disk with',checkpoint_dir)
        torch.save({
            "step": total_steps,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }, os.path.join(checkpoint_dir, f"{filename}_step_{total_steps}.pt"))     
    return total_steps



def compute_logprobs(model,dataloader,device,pad_token_id,eos_token_id,num_helper_samples=5,optimizer_helper=None,helper=None,move_to_device=False,max_num_steps=None,dataloader_test=None,horizon=1,gradient_accumulation_steps=8,num_epochs=3,num_test_batches=30,print_per_batch=10,batch_size=8,lah=0,temp=0.001,max_new_tokens=256,tokenizer=None,do_sample=False,with_ddp=True,ddp_rank=0,ddp_world_size=1,device_type='cuda'):
    losses=[]
    gradient_accumulation_steps=1
    total_steps=0
    total_processes_samples=0
    batch_size=dataloader.batch_size
    max_num_steps=max_num_steps//batch_size
    model.train()
    num_processed_tokens=[]
    for epoch in range(num_epochs):  # 3 epochs
        epoch_steps=0
        model.train()
        epoch_loss = 0
        loss_local = torch.tensor(0.0, device=device)

        if ddp_rank==0:
            print('-------------',len(dataloader))
        t0=time.time()
        for idx, batch in enumerate(dataloader):
            epoch_steps+=1
            B,T=batch['input_ids'].shape
            num_processed_tokens.append((B*T)*ddp_world_size)
            total_steps += 1
            
            if move_to_device:
#                device=model.device
#                print('shape',batch['input_ids'].shape,'attn',batch['attention_mask'].shape)
                inputs = batch['input_ids'][:,:].to(device)
                attention_mask = batch['attention_mask'][:,:].to(device)
                labels = batch['labels'][:,:].to(device)
            else:
                inputs = batch['input_ids'][:,:]
                attention_mask = batch['attention_mask'][:,:]
                labels = batch['labels'][:,:]
            if idx % print_per_batch == 0 and ddp_rank==0:
                if idx == 0:
                    print('???', idx, batch['input_ids'].shape)
                if idx > 0:
                    print('id:', idx, batch['input_ids'].shape, sum(losses[-print_per_batch:]) / print_per_batch,'total processes samples',total_processes_samples)
#            batch_size=inputs.shape[0]
            total_processes_samples+=batch_size

            current_length=inputs.shape[-1]#$np.min([outputs.shape[-1],inputs.shape[-1]])

            inputs = inputs[:,:current_length]#outputs
            attention_mask=attention_mask[:,:current_length]
#            labels[:,current_length:]=-1
            labels=labels[:,:current_length]
            
            # Assuming `outputs` is the model output
            if lah>0:
                last_index=torch.sum(attention_mask,dim=-1)-1
                B=attention_mask.shape[0]
#                non_pad_mask = inputs != pad_token_id
                seq_len = labels.size(1)  # Maximum sequence length
                range_tensor = torch.arange(seq_len, device=labels.device).unsqueeze(0)  # Shape: (1, seq_len)

                # Broadcast and compare against `last_index`
                # Mask tokens that are beyond the last valid index or are -100 in the labels
                answer_mask = (range_tensor <= last_index.unsqueeze(1)) & (labels != -100)
                question_mask = (range_tensor <= last_index.unsqueeze(1)) & (labels == -100)
                last_index=torch.sum(answer_mask,dim=-1)-1
                last_question_index=torch.sum(question_mask,dim=-1)
                answer_ids = torch.masked_select(inputs, answer_mask)
                num_el=256#answer_ids.shape[0]#labels[0:,last_question_index:].shape[-1]
                min_len=10#inputs.shape[-1]-5#int(inputs.shape[-1]-last_question_index)
          #      max_len=inputs.shape[-1]
                outputs = model.generate(input_ids=inputs[0:,:last_question_index].to(device),attention_mask=attention_mask[0:,:last_question_index].to(device), temperature=temp,pad_token_id=pad_token_id, top_p=0.9,top_k=50,min_length=min_len, max_new_tokens=max_new_tokens,return_dict_in_generate=True,output_scores=True,do_sample=do_sample)
           #     outputs = model.generate(input_ids=inputs[0:,:last_question_index].to(device),attention_mask=attention_mask[0:,:last_question_index].to(device),early_stopping=early_stopping, temperature=temp,pad_token_id=pad_token_id, top_p=0.9,top_k=50,min_length=min_len, max_length=max_len,return_dict_in_generate=True,output_scores=True,do_sample=do_sample)
#                                                                                   outputs_gen = model.generate(input_ids=inputs, attention_mask=attention_mask,early_stopping=early_stopping, temperature=temp,pad_token_id=tokenizer.pad_token_id, top_p=0.9,top_k=50, max_new_tokens=max_new_tokens)
                if False:
                    for (word_out,word_in) in zip(inputs[0],outputs['sequences'][0]):
                        if word_in!=word_out and word_in!=-100:
                            print('compare',tokenizer.convert_ids_to_tokens([word_out]),tokenizer.convert_ids_to_tokens([word_in]))
                answer_length=outputs['sequences'].shape[1]
                input_length=inputs.shape[1]
                logits=torch.stack(outputs['scores'],dim=1).to(device)
                num_gen_elements=logits.shape[1]
                final_index=min([last_question_index+num_el,inputs.shape[1]])
                if answer_length>input_length:
                    print('padding to the input required')
#                    print('labels after last_question_index:',labels[0:,last_question_index:])
                    pad_length=answer_length-input_length
                    eos_padding = torch.full((batch_size, pad_length), eos_token_id, dtype=torch.long, device=outputs['sequences'].device)
                    inputs= torch.cat([inputs, eos_padding], dim=1)
                    labels= torch.cat([labels, eos_padding], dim=1)
#                    print('new shapes are',inputs.shape,labels.shape,outputs['sequences'].shape)
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print(tokenizer.decode(inputs[0,last_question_index:]))
                    print(tokenizer.decode(outputs['sequences'][0,last_question_index:]))
           #     pad_length = last_question_index+num_el - outputs['sequences'].shape[1]
                elif answer_length<input_length:
                    print('padding to the output required')
                    print('===================================')
                    pad_length=input_length-answer_length
                    eos_padding = torch.full((batch_size, pad_length), eos_token_id, dtype=torch.long, device=outputs['sequences'].device)
                    outputs['sequences'] = torch.cat([outputs['sequences'], eos_padding], dim=1)
                    print(tokenizer.decode(inputs[0,last_question_index:]))
                    print(tokenizer.decode(outputs['sequences'][0,last_question_index:]))


           #     inputs[0:,:final_index]=outputs['sequences'][:,:final_index]                
                outputs = model(input_ids=inputs, labels=labels,attention_mask=attention_mask)
                loss = outputs.loss / gradient_accumulation_steps
            else:
 #               ddp_debug()
#                torch.cuda.empty_cache()
              #  inputs.requires_grad_(True)
                with torch.inference_mode():
                    print(device,inputs.device,labels.device,model.device,attention_mask.device)
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        outputs = model(input_ids=inputs, labels=labels,attention_mask=attention_mask)
                        loss = outputs.loss / gradient_accumulation_steps
               # print("Gradient norm2:", grad2.norm().item() if grad2 is not None else "None")
#                 import code;code.interact(locals=locals())
    #            print('loss.shape',loss.shape,loss,'inputs',inputs.shape)
  #          grad = model.module.model.layers[-1].self_attn.q_proj.weight.grad

            epoch_loss += loss.mean().item() * gradient_accumulation_steps  # Multiply back to the original scale
            losses.append(loss.mean().item() * gradient_accumulation_steps)
            if (max_num_steps is not None) and (total_steps>=max_num_steps):
                break

        avg_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch + 1}/{num_epochs}, train Loss: {avg_loss})")#, test loss:{loss_eval}")
    return total_steps
def generate_prompt_from_problem_codeforces(problem, lang='cpp', with_reasoning=True):
#    print(problem)
#    print('inside _codeforces')
    if 'language' in problem:
        lang=problem['language']
    assert lang in ['cpp','python']
#    prompt = f"// Problem: {problem['title']}\n"
#for ex in dataset2_train[0]['examples']:
    sys_prompt=coding_prompts[lang]
    user_prompt = f"// Problem description:\n{problem['description']}\n"
    user_prompt += f"\n// Input format:\n{problem['input_format']}\n"
    user_prompt += f"// Output format:\n{problem['output_format']}\n"
    user_prompt += "\n// Examples:\n"

    for ex in problem['examples']:
     #   prompt += f"// Input:\n{ex[0]}\n// Output:\n{ex[1]}\n"
        user_prompt+= f"\nInput\n{ex['input']}\nOutput\n{ex['output']}"
    #user_prompt += f"// Notes:\n{problem['note']}\n"
    
    if lang == "python":
        sys_prompt += "Do not import the same module more than once.\n"
        sys_prompt += "\nYour response must end with \n"
        sys_prompt += """
if __name__ == "__main__":
    solve()
"""
    elif lang == "cpp":
#        prompt += "\n// Please write a complete C++ program that reads from standard input and writes to standard output.\n"
        pass
#        sys_prompt += "// You may use STL. Avoid using any non-standard libraries.\n"

    if with_reasoning:
        sys_prompt += "Your answer must be structured as follows:\n"
        sys_prompt += "### Reasoning:<think> step by step reasoning here </think>\n"
        sys_prompt += f"### Code: ```{lang}\n<your code here>\n```\n"
        sys_prompt += "Your response ends with the code. Do not include any explanation or additional formatting after the code ends.\n"
    else:
        sys_prompt += f"Do not include any explanation or markdown formatting. Only output valid {lang} code.\n"

    return  {'system':sys_prompt,'user':user_prompt}


def evaluate_problem(problem, model,with_reasoning=True, k=1, temperature=0.8,return_logprobs=False,lang='cpp',mode='codeforces',VERBOSE=False):
#    print('mode is',mode)
    prompt = generate_prompt_from_problems(problem,lang=lang,with_reasoning=with_reasoning,mode=mode)[0]
    if 'problem_id' in problem.keys():
        problem_key='problem_id'
    else:
        problem_key='id'
#    print(f"\n\n--- Prompt for problem {problem.get(problem_key, '?')} ---\n{prompt}")

    tests = problem.get('official_tests') or problem.get('examples', [])
    total = len(tests)
    results = []
    # Try up to k samples
    if return_logprobs:
        completions,logprobs,stats = model(prompt, k=k, temperature=temperature) if k > 1 else [model(prompt, k=1, temperature=temperature)]        
    else:
        completions = model(prompt, k=k, temperature=temperature) if k > 1 else [model(prompt, k=1, temperature=temperature)]
#    print('completions have length',len(completions))
    for i, code in enumerate(completions):
        
 #       print(f"\n--- Code Generated (sample {i+1}) ---\n{code}\n")
        if isinstance(code, list):  # defensive programming
  #          print('len(code)',len(code))
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
            if VERBOSE:
                print(f"code does not compile")
            actual = "ERROR"                
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
                if VERBOSE:
                    print('actual output:', actual)
                    print('expected output:', expected)
                if actual == expected:
                    passed += 1
                else:
                    break
        results.append({
            "problem_id": problem[problem_key],
            "sample_id": i + 1,
            "passed": passed,
            "total": total,
            "success": passed == total
        })

        if passed == total:
            print(f"✅ Solved {problem[problem_key]} with sample {i+1}/{k}")
            break
        
    all_results= {
        "problem_id": problem[problem_key],
        "samples": results  # return all attempted completions
    }
    return all_results if all_results else {
        "problem_id": problem[problem_key],
        "sample_id": None,
        "passed": 0,
        "total": total,
        "success": False
    }

def tokenize_codeforce(examples, tokenizer,context_length=16834,include_assistant=True,truncation=False,padding="max_length"):

    messages=[[]]
    idx_kk=0
    msg_in=[]
    prompt=[]
    # Create the pre-context with helper function and add the new user content
    problems=examples
    structured_problem=generate_prompt_from_problems(problems, lang='cpp', with_reasoning=True)[0]
#    print(structured_problem)
    
    msg_in.append({"role": "system", "content": structured_problem['system']})
    msg_in.append({"role": "user", "content": structured_problem['user']})
    if include_assistant:
        msg_in.append({"role": "assistant", "content": examples['assistant']})
    else:
#        print('we are here')
        msg_in.append({"role": "assistant", "content": ''})

    prompt.append({"role": "system", "content": structured_problem['system']})
    prompt.append({"role": "user", "content": structured_problem['user']})
    messages[0]=msg_in
    # Apply chat template and tokenize
#    prompt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=True)
    encoded_inputs = tokenizer([
        tokenizer.apply_chat_template(msg, add_generation_prompt=False, tokenize=False) for msg in messages],
        padding=padding,#True,
        max_length=context_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
#    print(encoded_inputs['input_ids'].shape)
    ideal_outputs = encoded_inputs

    labels = ideal_outputs['input_ids'].clone()
    attention_mask=ideal_outputs['attention_mask'].clone()
    # This is the newly generated assistant message; unmask these tokens
#    start_idx = encoded_prompt['input_ids'].shape[-1] remove and infer it from padding mask of prompt instead

    encoded_prompt = tokenizer(
        tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False),
        padding=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    prompt_ids=encoded_prompt['input_ids'].clone()
    pad_token_id=tokenizer.pad_token_id
    non_pad_input_mask = prompt_ids != pad_token_id
#    non_pad_label_mask = labels != pad_token_id
    B=encoded_inputs['input_ids'].shape[0]
    max_non_pad_idx=torch.sum(non_pad_input_mask,dim=-1)-1#torch.max(torch.sum(non_pad_input_mask,dim=-1),torch.sum(non_pad_label_mask,dim=-1))-1

    shifted_idx=max_non_pad_idx#int(start_idx)
    for k in range(B):
        labels[k, :shifted_idx[k]] = -100
#        labels[k, shifted_idx[k]:] = ideal_outputs['input_ids'].clone()[k, shifted_idx[k]:]
#        print('att',attention_mask.shape)
        last_token=torch.sum(attention_mask[k,:],dim=-1)
        labels[k,last_token:]=-100
    inputs = {'input_ids': encoded_inputs['input_ids'],'labels':labels,'attention_mask':encoded_inputs['attention_mask'],'encoded_prompt':encoded_prompt}

    return inputs

  
def generate_prompt_from_problem_codelo(problem, lang='cpp', with_reasoning=True):
    print(problem)
#    prompt = f"// Problem: {problem['title']}\n"
    prompt = f"// Problem description:\n{problem['description']}\n"
    prompt += f"\n// Input format:\n{problem['input']}\n"
    prompt += f"// Output format:\n{problem['output']}\n"
    prompt += "\n// Examples:\n"

    for ex in problem['examples']:
        prompt += f"// Input:\n{ex[0]}\n// Output:\n{ex[1]}\n"

    if lang == "python":
        prompt += "\n// Please write a function named `solve()` that reads from standard input and writes to standard output. Your code can start by reading the input stream"
        prompt += """\n
def solve():
    import sys
    <import other needed moduels>
    input = sys.stdin.read
"""
        prompt += "Do not import the same module more than once.\n"
        prompt += "\nYour response must end with \n"
        prompt += """
if __name__ == "__main__":
    solve()
"""
    elif lang == "cpp":
        prompt += "\n// Please write a complete C++ program that reads from standard input and writes to standard output.\n"
        prompt += "// You may use STL. Avoid using any non-standard libraries.\n"

    if with_reasoning:
        prompt += "Your answer must be structured as follows:\n"
        prompt += "### Reasoning:<think> step by step reasoning here </think>\n"
        prompt += f"### Code: //your {lang} code here\n"
        prompt += "Do not include any explanation or additional formatting after the code ends.\n"
    else:
        prompt += f"Do not include any explanation or markdown formatting. Only output valid {lang} code.\n"

    return prompt
#def query_qwen_model(prompt):
    # Replace with your inference pipeline or local API
 #   raise NotImplementedError("Connect this to your Qwen-7B model")


def generate_prompt_from_problems(problems, lang='cpp', with_reasoning=True,mode='codeforces'):
    results=[]
    first_key=list(problems.keys())[0]
    number_of_problems=len(problems[first_key])
    if type(problems[first_key])==list:
      #  print(problems)
        for i in range(number_of_problems):
            problem={k: v[i] for k, v in problems.items()}
          #  print('problem',problem)
            if mode=='codeforces':
                results.append(generate_prompt_from_problem_codeforces(problem, lang, with_reasoning))
            else:
                results.append(generate_prompt_from_problem_codelo(problem, lang, with_reasoning))
    else:
        if mode=='codeforces':
           # print('were here')
            results=[generate_prompt_from_problem_codeforces(problems, lang, with_reasoning)]        
        else:
            results=[generate_prompt_from_problem_codelo(problems, lang, with_reasoning)]        
    return results

def get_prompt_m1_default(instruct=None,Tree=None,icl_examples=None):

    if instruct is None:
        prompt = [
        {"role": "system", "content": f"You are a helpful coding assistant"""},
        ]
    else:
#        print('instruct to m1_restricted is',instruct)
        prompt=[{"role": "system", "content":instruct}]
    if icl_examples is not None:
        prompt.extend(icl_examples)
    return prompt

def my_apply_chat_template(msg, add_generation_prompt=False, tokenize=False):
    instruct=''
    for m in msg:
        if m['role'].lower()=='system':
            instruct+=''+m['content']+'<|eot|>'
        elif m['role'].lower()=='user':
            instruct+='<|bos|>'+ 'User:'+m['content']+'<|eot|>'
        else:
            instruct+='<|bos|>'+'Assistant:'+m['content']+'<|eot|>'
    if add_generation_prompt:
        instruct+='<|bos|>'+'Assistant:'
    return instruct

def get_llm_output_on_text_batch(model,data,tokenizer,device,mode='default',verbose=False,print_rel=True,return_output_attentions=False, return_embeddings=False,return_hidden_states=False,instruct_flag=True,history=None,use_input_embeddings=False,instruct=None,examples=None, append_to_answer=' ', start_idx=0,end_idx=None,micro_batch_size=8,temp=0.001,incontext_mode=-1,moveflag=True,print_every=10,Tree=None,knowledge_tree=None,topk=1,at_k=1,**kwargs):
    print('data has type',type(data),len(data))
    if type(data)==type('str'):
        data=[{'text':data}]
    elif type(data)==list and type(data[0])==type('str'):
        data=[{'text':entry} for entry in data]
    elif type(data)==list and type(data[0])==dict:
        #assuming each data entry is {'system message':'content','user message':content}
        keys=list(data[0].keys())
      #  print('keys are',keys)
        data=[{'text':entry[keys[1]],'instruct':entry[keys[0]]} for entry in data]
    else:
        raise NotImplementedError("data type not recognized")
    # Replicate each item `at_k` times
   # print(data[0])
    expanded_data = []
    for entry in data:
        expanded_data.extend([entry] )
    data=expanded_data        
    # Option 1: Check the actual token string
    eos_token= tokenizer.eos_token

    DEFAULT_GENERATE_CONFIG = {
        "top_p": 0.8,                    # Nucleus sampling: filters unlikely tokens
        "top_k":20,
        "num_beams":1,
        "num_return_sequences": at_k,
        "do_sample": True,               # Enables sampling instead of greedy decoding
        "temperature":temp,
        "max_new_tokens":256,
        "early_stopping":True
    }
    if DEFAULT_GENERATE_CONFIG.get('num_beams',1)==1:
        DEFAULT_GENERATE_CONFIG.pop('early_stopping',None)

    generate_config = {**DEFAULT_GENERATE_CONFIG, **kwargs}
#    print('-------------------max new tokens',generate_config["max_new_tokens"])
#    print('--------------------------------------------------------generate config',generate_config)
 #   print('current dataset has len',len(data),'at_k',at_k)
    if verbose:
        print('starting with',len(data),'samples')   
    if end_idx==None:
        end_idx=len(data)
        data=data[start_idx:]
    else:
        end_idx=min([len(data),end_idx])
        data=data[start_idx:end_idx]
    results=[[]]*(len(data)*at_k)

    end_idx=end_idx-1
    for key in data[0].keys():
        if key=='text' or key=='full_text':
            text_key=key
    if verbose:
        print('current dataset has len',len(data))
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side = 'left'

    function_name='get_prompt_'+incontext_mode_to_key[incontext_mode]+'_'+mode
#    print(function_name)
    function_to_call = globals()[function_name]

    msg=function_to_call(instruct=instruct,icl_examples=examples)
    if instruct_flag:
        text_precontext=tokenizer.apply_chat_template(msg, add_generation_prompt=False, tokenize=False)
    else:
        text_precontext=my_apply_chat_template(msg, add_generation_prompt=False, tokenize=False)



    max_id=int((1+int(end_idx/micro_batch_size))*micro_batch_size)
    #data=dataset
    for idx in range(0,max_id,micro_batch_size):
    #for idx in range(0,85):#,micro_batch_size):#len(data),micro_batch_size):
        if idx%10==0:
            if verbose:
                print('processing sample number ',idx)
        adjusted_size=micro_batch_size
        if len(data)<idx+micro_batch_size:
            adjusted_size=len(data)-idx
        #print(prompts)
        messages=[[]]*adjusted_size
        marked_texts=[[]]*adjusted_size
        for idx_kk in range(adjusted_size):
            if not 'instruct' in data[idx+idx_kk]:
                instruct_idx=instruct
            else:
                instruct_idx=data[idx+idx_kk]['instruct']
                
            if idx_kk+idx>=len(data):
                continue        

            if 'full_text' in data[0].keys():
                text_key='full_text'
                text=data[idx+idx_kk][text_key]
                fact=''
                if knowledge_tree is not None:
                    fact=knowledge_tree[text]
                    if fact=='NONE':
                        fact=''
                    else:
                        fact=fact.tag()
                                                    
                msg.append({"role": "user", "content":text})
            else:
                text_key='text'
#                print('------------',data[idx+idx_kk])
                text=data[idx+idx_kk][text_key]
                facts=''
                if knowledge_tree is not None:
                    facts=knowledge_tree[(text,topk)]
                    if facts=='NONE':
                        facts=''
                fact=''                
                if type(facts)==list:
                    for fac in facts:
                        if fac.tag('cap')!='NONE':
                            fact+=fac.tag('cap')+'\n'
                    if verbose:
                        print('relevant facts',fact)
                else:
                    if verbose:
                        print('facts are',facts,'fact is',fact,facts=='')
                    if facts!='' and facts.tag('cap')!='NONE':
                        fact=facts.tag('cap')
#                print('----------------------------------------------------')
                if fact!='' and fact!='NONE':
                    instruct_idx=instruct_idx+'\n'+fact
                msg=function_to_call(instruct=instruct_idx,icl_examples=examples)
                if verbose:
                    print('===============text is',text)
                    print('===============added fact is',fact)
                    print('===============instruct_idx is',instruct_idx)
                    print('===============instruct is',instruct)


                msg.append({"role": "user", "content":text})
                    


            messages[idx_kk]=msg
            marked_texts[idx_kk]=text
        append_to_answers=[append_to_answer]*adjusted_size
        if instruct_flag:
            encoded_inputs = tokenizer(
            [tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)+app.strip() for (msg,app) in zip(messages,append_to_answers)],
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
            )
        else:
            encoded_inputs = tokenizer(
            [my_apply_chat_template(msg, add_generation_prompt=True, tokenize=False)+app.strip() for (msg,app) in zip(messages,append_to_answers)],
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
            )

        # Extract input_ids and attention_mask
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        common_index=len(text_precontext)#+len('<|start_header_id|>user<|end_header_id|>')
        return_dict_in_generate=return_output_attentions or return_hidden_states
#        print(len(input_ids),type(input_ids))
#        input_ids = input_ids.repeat_interleave(at_k, dim=0)  # (B*k, L)
#        B=input_ids.shape[0]
        if verbose:
            print('preping for model call',input_ids.shape,attention_mask.shape)
        if use_input_embeddings:
        # Generate the output
            with torch.no_grad():
                outputs = model.generate(input_embeds=input_embeddings, output_logits=True,        return_dict_in_generate=return_dict_in_generate,output_attentions=return_output_attentions, output_hidden_states=return_hidden_states,attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, **generate_config)
        else:
            with torch.no_grad():
                outputs = model.generate(input_ids=input_ids, output_logits=False,return_dict_in_generate=return_dict_in_generate,output_attentions=return_output_attentions, output_hidden_states=return_hidden_states,attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id, **generate_config)
        if verbose:
            print('output reshaped successfully',outputs.shape)
   #     print('model call finished successfully',outputs.shape,outputs.dtype)
#        print(outputs[0])
#        outputs = outputs.view(B, at_k, -1)

        if hasattr(outputs, "sequences"):
            generated_tokens = outputs.sequences
        else:
            generated_tokens = outputs

#        print(return_hidden_states,outputs)

        output_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if verbose:
            print('output_texts computed with shape',len(output_texts))
            print('output_text is',output_texts)


        start_indices=[common_index+len(text) for text in marked_texts]#*adjusted_size
        use_extra=False
        outputs_extra=''
        if return_hidden_states or return_output_attentions:
            if use_extra:
                outputs_extra = model(generated_tokens, output_attentions=return_output_attentions, output_hidden_states=return_hidden_states)
                if return_output_attentions:
                    attentions = outputs_extra.attentions[-1][:,common_index:].float()  
                    attn_avg = attentions.mean(dim=1)[0].detach().cpu().numpy()  # shape: (seq_len, seq_len)
            else:
                target_layer=-1
                init_hidden_states = outputs.hidden_states[0][target_layer][:, :-1, :]  # exclude the first generated token

                # 2. Collect newly generated hidden states at each step
                new_hidden_states = []

                for step_idx, step_hidden_states in enumerate(outputs.hidden_states):
                    # At each step, pick the last token only
                    h = step_hidden_states[target_layer][:, -1:, :]  # (batch_size, 1, hidden_dim)
                    new_hidden_states.append(h)

                # 3. Concatenate initial + new generated
                full_sequence_hidden_states = torch.cat([init_hidden_states] + new_hidden_states, dim=1)
                full_logits = torch.cat([logit.unsqueeze(1) for logit in outputs.logits], dim=1)

                outputs_extra = CausalLMOutput(
                    logits=full_logits,          # if you don't have logits, set to None
                    hidden_states=(full_sequence_hidden_states,),  # must be a tuple or list
                    attentions=None,      # optional
                )
#        print('all outputs',output_texts)
   #     print('output is',outputs.shape,outputs[0,326:],flush=True)
        for kk in range(adjusted_size):  # adjusted_size = micro_batch_size
            base_idx = idx + kk  # current example index in original data

            if base_idx >= len(data):
                continue

            for k in range(at_k):  # iterate over the k samples per input
                output_index = kk * at_k + k  # compute the correct local output index
           #     print('output_index',output_index,'len(text)',len(output_texts),'kk',kk,'k',k,'base index',base_idx,'idx',idx)
                if output_index >= len(output_texts):
                    continue

                output_text = output_texts[output_index]
                text = data[base_idx][text_key]

                # Use precomputed start index for this input
                start_index = start_indices[kk]
                pre_relevant_output = output_text[start_index:]

                # Locate start of assistant response
                start_token = '>assistant'
                start_idx = pre_relevant_output.find(start_token)
                if start_idx != -1:
                    relevant_output = pre_relevant_output[start_idx + len(start_token):]
                else:
                    relevant_output = pre_relevant_output  # fallback if token not found

                # Cut off at <|eot_id|> if present
                end_idx = relevant_output.find(eos_token)
                if end_idx > 0:
                    relevant_output = relevant_output[:end_idx]
           #     print('adding relevant output of len',len(relevant_output))
                global_output_index=idx*at_k+output_index
          #      print('global output_index',global_output_index)
                results[global_output_index]=relevant_output

                if print_rel:
                    print(f"[{base_idx}]@{k}: {relevant_output}")

    if return_embeddings:
        model_embeder=model.get_input_embeddings()
        input_embeddings=model_embeder(input_ids)#model.model.embed_tokens
        return results,outputs_extra,input_embeddings#,input_ids,attention_mask

    else:
        return results,outputs_extra
import os, re
from typing import Optional, Tuple

_STEP_RE  = re.compile(r"(?:^|[_-])step[_-]?(\d+)\.pt$", re.IGNORECASE)
_EPOCH_RE = re.compile(r"(?:^|[_-])epoch[_-]?(\d+)$", re.IGNORECASE)
_DIGITS_RE = re.compile(r"^\d+$")


def _maybe_strip_module(state_dict):
    """If keys start with 'module.', strip them."""
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def _maybe_add_module(state_dict):
    """If keys do NOT start with 'module.', add them. Useful if model is wrapped in DDP."""
    if not state_dict:
        return state_dict
    if all(not k.startswith("module.") for k in state_dict.keys()):
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict

def _parse_epoch(name: str) -> Optional[int]:
    m = _EPOCH_RE.search(name)
    if m:
        return int(m.group(1))
    if _DIGITS_RE.match(name):
        return int(name)
    return None

def _best_pt_in_dir(d: str) -> Tuple[Optional[str], int]:
    """Return (best_ckpt_path, best_step) among *.pt directly in dir d."""
    best_file, best_step = None, -1
    for fname in os.listdir(d):
        if not fname.endswith(".pt"):
            continue
        m = _STEP_RE.search(fname)
        step = int(m.group(1)) if m else -1
        if step > best_step:
            best_step, best_file = step, fname
    return (os.path.join(d, best_file) if best_file else None, best_step)

def _find_latest_pt(path: str) -> Optional[Tuple[str, int]]:
    """
    If `path` is a file -> return (path, step).
    If `path` is a dir that already contains *.pt -> pick the largest step there.
    Else treat `path` as ROOT = model_name/dataset, find the epoch dir with highest number,
    then pick the largest step inside that epoch dir.
    Returns (ckpt_path, epoch_num). epoch_num = -1 if not parsable.
    """
    if os.path.isfile(path):
        # parse step from filename for completeness
        m = _STEP_RE.search(os.path.basename(path))
        step = int(m.group(1)) if m else -1
        # parse epoch from parent dir
        parent = os.path.basename(os.path.dirname(path))
        epoch_num = _parse_epoch(parent)
        return path, (epoch_num if epoch_num is not None else -1)

    if not os.path.isdir(path):
        return None

    # Case 1: directory directly holds .pt files
    ckpt, _ = _best_pt_in_dir(path)
    if ckpt:
        parent = os.path.basename(os.path.dirname(ckpt))
        epoch_num = _parse_epoch(os.path.basename(path)) or _parse_epoch(parent) or -1
        return ckpt, epoch_num

    # Case 2: treat as ROOT holding epoch subdirs
    best_epoch_num, best_epoch_dir = -1, None
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if not os.path.isdir(full):
            continue
        ep = _parse_epoch(name)
        if ep is None:
            continue
        if ep > best_epoch_num:
            best_epoch_num, best_epoch_dir = ep, full

    if best_epoch_dir is None:
        return None

    ckpt, _ = _best_pt_in_dir(best_epoch_dir)
    if ckpt:
        return ckpt, best_epoch_num

    # Optional: one more level (epoch/.../*.pt), e.g. epoch/ model_steps/
    for sub in os.listdir(best_epoch_dir):
        subdir = os.path.join(best_epoch_dir, sub)
        if os.path.isdir(subdir):
            ckpt, _ = _best_pt_in_dir(subdir)
            if ckpt:
                return ckpt, best_epoch_num

    return None

from typing import Optional, Tuple
import torch

def load_checkpoint_if_any(
    resume_path: Optional[str],
    model=None,
    optimizer=None,
    scheduler=None,
    map_location=None,
    ddp_wrapped: bool = False,
    strict: bool = True,
) -> Tuple[int, int, Optional[str]]:
    """
    Returns:
      (start_step, epoch_num, loaded_path)
      start_step = 0 and epoch_num = -1 if nothing loaded.
    """
    if resume_path is None:
        return 0, 0, None

    found = _find_latest_pt(resume_path)
    if found is None:
        print(f"[resume] WARNING: nothing found at {resume_path}")
        return 0, -1, None

    ckpt_path, epoch_num = found
    print(f"[resume] Loading checkpoint: {ckpt_path} (epoch={epoch_num})")
    ckpt = torch.load(ckpt_path, map_location=map_location)

    saved_step = int(ckpt.get("step", 0))
    # prefer explicit epoch in checkpoint if present
    epoch_num = int(ckpt.get("epoch", epoch_num))

    sd = ckpt.get("model_state_dict", {})
    if ddp_wrapped:
        target = model.module
        try:
            target.load_state_dict(_maybe_strip_module(sd), strict=strict)
        except RuntimeError:
            target.load_state_dict(_maybe_add_module(sd), strict=strict)
    else:
        try:
            model.load_state_dict(_maybe_strip_module(sd), strict=strict)
        except RuntimeError as e:
            print(f"[resume] strict={strict} failed once, retrying with module prefix. Err: {e}")
            model.load_state_dict(_maybe_add_module(sd), strict=False)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[resume] WARNING: optimizer state not loaded cleanly: {e}")

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"[resume] WARNING: scheduler state not loaded cleanly: {e}")

    print(f"[resume] Loaded step={saved_step}, epoch={epoch_num}")
    return saved_step, epoch_num, ckpt_path
