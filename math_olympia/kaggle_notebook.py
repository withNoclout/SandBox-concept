# --- KAGGLE SUBMISSION NOTEBOOK ---
# Copy and paste this entire script into a code cell in your Kaggle Notebook.
# Make sure you have added the model dataset (e.g., deepseek-math-7b-rl) to your notebook.

import os
import sys
import subprocess
import signal
import contextlib
import io
import re
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. CONFIGURATION ---
# Path to your model on Kaggle (Update this!)
# WARNING: 30B model requires ~30GB VRAM. T4 x2 has 32GB. This is very tight.
MODEL_PATH = "/kaggle/input/qwen-3/transformers/30b-a3b-thinking-2507-fp8/1" 
# If using Qwen: "/kaggle/input/qwen2.5-math-7b/Qwen2.5-Math-7B-Instruct"

# --- 2. CACHE MODEL (Speed Optimization) ---
def cache_model(path, exts=(".bin", ".pt", ".safetensors", ".model"), num_workers=None, chunk_mb=256):
    """Pre-read model weight files into OS page cache."""
    print(f"[cache_model] Warming up cache for {path}...")
    
    def warmup_file(fpath):
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        with open(fpath, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                total += len(data)
        return fpath, total

    if os.path.isdir(path):
        files = [
            os.path.join(root, name)
            for root, _, names in os.walk(path)
            for name in names
            if name.endswith(exts)
        ]
        files.sort()
    else:
        files = [path]

    if not files:
        print(f"[cache_model] Warning: No model files found in {path}")
        return 0

    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    t0 = time.time()
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            # print(f"[{i}/{len(files)}] cached {os.path.basename(fpath)}")

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    print(f"[cache_model] Read {gb:.2f} GB in {elapsed:.2f}s")
    return total_bytes

# Run cache
try:
    cache_model(MODEL_PATH)
except Exception as e:
    print(f"Cache failed (non-fatal): {e}")

# --- 3. MODEL LOADING (Offline) ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"CRITICAL ERROR Loading Model: {e}")
    # Fallback for testing without model
    model = None
    tokenizer = None

# --- 4. SOLVER LOGIC (TIR) ---
SYSTEM_PROMPT = """You are an expert mathematician and Python programmer.
Your goal is to solve complex mathematical problems by writing and executing Python code.

Instructions:
1. Analyze the problem.
2. Write Python code to calculate the answer.
3. Verify the answer if possible.
4. PRINT the final answer using: print(f"BOXED_ANSWER: {answer}")
5. The answer must be a non-negative integer (0-99999).
"""

# --- 6. SUBMISSION LOOP (With Majority Voting) ---
import kaggle_evaluation.aimo_3_inference_server
import polars as pl
from collections import Counter

# Configuration
N_ATTEMPTS = 8  # How many times to try each problem (Higher = Slower but Better)
TEMPERATURE = 0.6 # Add randomness for variety

def generate_solution_stochastic(problem_text):
    if model is None:
        return "print(f'BOXED_ANSWER: 0')"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this problem:\n{problem_text}"}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=TEMPERATURE, # Randomness enabled
            do_sample=True,
            top_k=40,
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def solve_problem(problem_text):
    """Generates one solution and extracts the answer."""
    response = generate_solution_stochastic(problem_text)
    code = extract_code(response)
    
    if not code:
        return None
        
    result = execute_python_code(code)
    
    try:
        # Parse Answer
        match = re.search(r'BOXED_ANSWER:\s*(\d+)', result)
        if match:
            return int(match.group(1)) % 100000
        else:
            numbers = re.findall(r'\d+', result)
            if numbers:
                return int(numbers[-1]) % 100000
    except:
        pass
    return None

def predict(test: pl.DataFrame, sample_submission: pl.DataFrame):
    print(f"Processing batch...")
    try:
        # Robust Data Access
        if isinstance(test, pl.DataFrame):
            problem_text = test.get_column("problem").item(0)
            problem_id = test.get_column("id").item(0)
        else:
            # Fallback for Pandas (if environment changes)
            problem_text = test['problem'].iloc[0]
            problem_id = test['id'].iloc[0]
    except Exception as e:
        print(f"CRITICAL ERROR accessing data: {e}")
        print(f"Type of test: {type(test)}")
        return sample_submission.with_columns(pl.lit(0).alias('answer'))
    
    print(f"Solving Problem {problem_id} (Voting {N_ATTEMPTS} times)...")
    
    answers = []
    
    # Try N times
    for i in range(N_ATTEMPTS):
        ans = solve_problem(problem_text)
        if ans is not None:
            answers.append(ans)
            print(f"  Attempt {i+1}: {ans}")
        else:
            print(f"  Attempt {i+1}: Failed")
    
    # Majority Vote
    final_answer = 0
    if answers:
        # Pick the most common answer
        counts = Counter(answers)
        final_answer = counts.most_common(1)[0][0]
        print(f"-> Selected Answer: {final_answer} (Votes: {counts})")
    else:
        print("-> No valid answers found. Guessing 0.")

    return sample_submission.with_columns(pl.lit(final_answer).alias('answer'))

# Start Server
server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    server.serve()
else:
    server.run_local_gateway(
        (
            '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',
        )
    )
