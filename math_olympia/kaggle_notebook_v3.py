# --------------------------------------------------------------
#  AI MATHEMATICAL OLYMPIAD – CHALLENGE 3 (Kaggle Notebook)
# --------------------------------------------------------------
#  This notebook is a rewrite of the original `kaggle_notebook_v2.py`
#  with the following improvements:
#   • Strict `BOXED_ANSWER:` extraction (no fallback to stray numbers)
#   • Deterministic model generation (temperature 0.3, fixed seed)
#   • Longer safe‑execution timeout (15 s) with the same alarm guard
#   • Detailed logging of each attempt (saved to `logs/` for debugging)
#   • Minor code‑style clean‑ups and defensive data handling
#   • All functionality kept in a single file for easy copy‑paste
# --------------------------------------------------------------

import os
import sys
import signal
import contextlib
import io
import re
import time
import json
import logging
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import polars as pl
import pandas as pd

# ------------------------------------------------------------------
# 1️⃣ CONFIGURATION
# ------------------------------------------------------------------
# Path to the model on Kaggle (adjust if you move the model)
MODEL_PATH = "/kaggle/input/qwen-3/transformers/30b-a3b-thinking-2507-fp8/1"

# Deterministic generation settings
TEMPERATURE = 0.3          # lower = more deterministic
MAX_NEW_TOKENS = 1024
TOP_K = 40
SEED = 42                 # reproducible runs

# Execution timeout (seconds) – long enough for most calculations
EXEC_TIMEOUT = 15

# Logging configuration (writes a tiny JSON line per attempt)
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "attempts.log",
    level=logging.INFO,
    format='%(asctime)s %(message)s',
)

# ------------------------------------------------------------------
# 2️⃣ MODEL LOADING (offline – no internet)
# ------------------------------------------------------------------
print("🔧 Loading model…")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    torch.manual_seed(SEED)          # set global seed for reproducibility
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Critical error loading model: {e}")
    model = None
    tokenizer = None

# ------------------------------------------------------------------
# 3️⃣ PROMPT / SYSTEM MESSAGE
# ------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert mathematician and Python programmer.
Your goal is to solve complex mathematical problems by writing and executing Python code.

Instructions:
1. Analyze the problem statement.
2. Write clean Python code that computes the answer.
3. Verify the answer if possible.
4. PRINT the final answer using exactly:
   print(f'BOXED_ANSWER: {answer}')
5. The answer must be a non‑negative integer in the range 0‑99999.
"""

# ------------------------------------------------------------------
# 4️⃣ HELPER: Strict BOXED_ANSWER extraction
# ------------------------------------------------------------------
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
    if 0 <= val <= 99999:
        return val % 100000
    return None

# ------------------------------------------------------------------
# 5️⃣ SAFE EXECUTION (timeout guard)
# ------------------------------------------------------------------
def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def execute_python_code(code: str, timeout: int = EXEC_TIMEOUT) -> str:
    """Run generated code in a sandboxed environment with a timeout."""
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

# ------------------------------------------------------------------
# 6️⃣ MODEL INFERENCE – generate solution (deterministic)
# ------------------------------------------------------------------
def generate_solution(problem_text: str) -> str:
    """Ask the model to solve a single problem and return its raw response."""
    if model is None:
        # Fallback stub – returns a dummy answer that will be filtered out later
        return "print(f'BOXED_ANSWER: 0')"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this problem:\n{problem_text}"},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=False,          # greedy decoding because temperature is low
        top_k=TOP_K,
    )
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def extract_code(response: str) -> str | None:
    """Pull the first fenced Python block from the model response."""
    m = re.search(r'```python(.*?)```', response, re.DOTALL)
    return m.group(1).strip() if m else None

# ------------------------------------------------------------------
# 7️⃣ SOLVE ONE PROBLEM (full pipeline)
# ------------------------------------------------------------------
def solve_problem(problem_text: str) -> int | None:
    """Run the whole chain: generate → extract code → execute → parse answer."""
    raw = generate_solution(problem_text)
    code = extract_code(raw)
    if not code:
        logging.info(json.dumps({"stage": "no_code", "problem": problem_text}))
        return None

    out = execute_python_code(code)
    answer = extract_boxed_answer(out)

    logging.info(
        json.dumps(
            {
                "stage": "attempt",
                "problem": problem_text,
                "code": code,
                "output": out,
                "boxed_answer": answer,
            }
        )
    )
    return answer

# ------------------------------------------------------------------
# 8️⃣ PREDICT FUNCTION – what Kaggle calls for each row
# ------------------------------------------------------------------
def predict(test, sample_submission):
    """
    Kaggle will invoke this for every test row.
    It returns a Polars DataFrame with columns `id` and `answer`.
    """
    # ----- Robust data extraction -----
    try:
        if isinstance(test, pl.DataFrame):
            problem_text = test.get_column("problem").item(0)
            problem_id = test.get_column("id").item(0)
        elif isinstance(test, pd.DataFrame):
            problem_text = test["problem"].iloc[0]
            problem_id = test["id"].iloc[0]
        else:  # very unlikely fallback
            problem_text = str(test.get("problem", ""))
            problem_id = str(test.get("id", ""))
    except Exception as e:
        print(f"⚠️ Data extraction error: {e}")
        return pl.DataFrame({"id": [0], "answer": [0]})

    print(f"🔎 Solving problem {problem_id} …")

    # ----- Majority voting (8 attempts) -----
    answers = []
    for attempt in range(8):
        ans = solve_problem(problem_text)
        if ans is not None:
            answers.append(ans)
            print(f"   ✅ attempt {attempt+1}: {ans}")
        else:
            print(f"   ❌ attempt {attempt+1} failed")

    final_answer = Counter(answers).most_common(1)[0][0] if answers else 0
    print(f"🏁 Final answer for {problem_id}: {final_answer}")

    submission_df = pl.DataFrame({"id": [problem_id], "answer": [int(final_answer)]})
    return submission_df

# ------------------------------------------------------------------
# 9️⃣ SERVER SETUP (Kaggle entry point)
# ------------------------------------------------------------------
import kaggle_evaluation.aimo_3_inference_server

server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    server.serve()
else:
    server.run_local_gateway(
        (
            "/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv",
        )
    )
