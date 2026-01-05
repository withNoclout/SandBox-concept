# --- KAGGLE SUBMISSION NOTEBOOK (Final Version) ---
# Copy and paste this entire script into a code cell in your Kaggle Notebook.
# Ensure the Qwen-30B model dataset is added as an input.

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
import kaggle_evaluation.aimo_3_inference_server
import polars as pl
import pandas as pd
from collections import Counter

# -------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------
# Path to the model on Kaggle (adjust if needed)
MODEL_PATH = "/kaggle/input/qwen-3/transformers/30b-a3b-thinking-2507-fp8/1"

# -------------------------------------------------------------------
# 2. OPTIONAL: CACHE MODEL WEIGHTS (speeds up first load)
# -------------------------------------------------------------------
def cache_model(path, exts=(".bin", ".pt", ".safetensors", ".model"), num_workers=None, chunk_mb=256):
    """Read model weight files into the OS page cache (optional)."""
    if not os.path.exists(path):
        print(f"Model path not found: {path}")
        return
    print(f"[cache_model] Warming up cache for {path}...")
    # Simple sequential read – good enough for Kaggle notebooks
    for root, _, names in os.walk(path):
        for name in names:
            if name.endswith(exts):
                fpath = os.path.join(root, name)
                try:
                    with open(fpath, "rb") as f:
                        while f.read(chunk_mb * 1024 * 1024):
                            pass
                except Exception as e:
                    print(f"Cache read error {fpath}: {e}")
    print("[cache_model] Done.")

# Uncomment the next line if you want to warm‑up the cache
# cache_model(MODEL_PATH)

# -------------------------------------------------------------------
# 3. MODEL LOADING (offline, no internet)
# -------------------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Model…")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",  # works on T4, H100, etc.
        trust_remote_code=True,
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Critical error loading model: {e}")
    model = None
    tokenizer = None

# -------------------------------------------------------------------
# 4. SOLVER LOGIC (Tool‑Integrated Reasoning)
# -------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert mathematician and Python programmer.
Your goal is to solve complex mathematical problems by writing and executing Python code.

Instructions:
1. Analyze the problem.
2. Write Python code to calculate the answer.
3. Verify the answer if possible.
4. PRINT the final answer using: print(f'BOXED_ANSWER: {answer}')
5. The answer must be a non‑negative integer (0‑99999).
"""

def generate_solution_stochastic(problem_text):
    """Generate a solution with a little randomness (temperature)."""
    if model is None:
        return "print(f'BOXED_ANSWER: 0')"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this problem:\n{problem_text}"},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.6,
            do_sample=True,
            top_k=40,
        )
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def extract_code(response):
    m = re.search(r'```python(.*?)```', response, re.DOTALL)
    return m.group(1).strip() if m else None

# -------------------------------------------------------------------
# 5. SAFE EXECUTION (7‑second timeout)
# -------------------------------------------------------------------
def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def execute_python_code(code, timeout=7):
    buf = io.StringIO()
    signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        with contextlib.redirect_stdout(buf):
            signal.alarm(timeout)
            exec(code, {"__name__": "__main__"})
            signal.alarm(0)
        return buf.getvalue().strip()
    except TimeoutError:
        return "Error: Execution timed out"
    except Exception as e:
        signal.alarm(0)
        return f"Error: {e}"

def extract_boxed_answer(output: str) -> int | None:
    """
    Search for the exact token ``BOXED_ANSWER: <integer>`` in the model output.
    Returns the integer (mod 100000) if found and within the allowed range,
    otherwise returns None.
    """
    m = re.search(r'BOXED_ANSWER:\s*(\d+)', output)
    if not m:
        return None
    val = int(m.group(1))
    # Ensure answer is within competition bounds (0‑99999)
    if 0 <= val <= 99999:
        return val % 100000
    return None

def solve_problem(problem_text):
    resp = generate_solution_stochastic(problem_text)
    code = extract_code(resp)
    if not code:
        return None
    out = execute_python_code(code)
    # Look for the boxed answer
    m = re.search(r'BOXED_ANSWER:\s*(\d+)', out)
    if m:
        return int(m.group(1)) % 100000
    # Fallback – grab the last integer we can find
    nums = re.findall(r'\d+', out)
    return int(nums[-1]) % 100000 if nums else None

# -------------------------------------------------------------------
# 6. PREDICT FUNCTION (what Kaggle calls for each row)
# -------------------------------------------------------------------
def predict(test, sample_submission):
    # ----- Robust data extraction -----
    try:
        if isinstance(test, pl.DataFrame):
            problem_text = test.get_column("problem").item(0)
            problem_id   = test.get_column("id").item(0)
        elif isinstance(test, pd.DataFrame):
            problem_text = test["problem"].iloc[0]
            problem_id   = test["id"].iloc[0]
        else:  # very unlikely, but keep a safe fallback
            problem_text = str(test["problem"]) if "problem" in test else ""
            problem_id   = str(test["id"])       if "id" in test else ""
    except Exception as e:
        print(f"⚠️ Data extraction error: {e}")
        # Return a dummy row so the pipeline keeps running
        return pl.DataFrame({"id": [0], "answer": [0]})

    print(f"🔎 Solving problem {problem_id} …")

    # ----- Majority voting (8 attempts) -----
    answers = []
    for _ in range(8):
        ans = solve_problem(problem_text)
        if ans is not None:
            answers.append(ans)
            print(f"   ✅ attempt: {ans}")
        else:
            print("   ❌ attempt failed")

    final_answer = Counter(answers).most_common(1)[0][0] if answers else 0
    print(f"🏁 Final answer for {problem_id}: {final_answer}")

    # ----- Build a clean submission DataFrame -----
    submission_df = pl.DataFrame({
        "id":     [problem_id],
        "answer": [int(final_answer)],
    })
    return submission_df

# -------------------------------------------------------------------
# 7. START THE SERVER (Kaggle will call this automatically)
# -------------------------------------------------------------------
server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Real competition run – block until Kaggle shuts us down
    server.serve()
else:
    # Local testing / notebook run – feed the test CSV directly
    server.run_local_gateway((
        '/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',
    ))
```
