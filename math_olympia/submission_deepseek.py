import kaggle_evaluation.aimo_3_inference_server
import polars as pl
import os
import sys
from deepseek_solver import solve_with_deepseek, extract_code
import io
import contextlib

# Get API Key from environment
API_KEY = os.environ.get("DEEPSEEK_API_KEY")

import signal

def handler(signum, frame):
    raise TimeoutError("Execution timed out")

def execute_python_code(code, timeout=5):
    """Executes the generated Python code with a timeout and captures the output."""
    output_buffer = io.StringIO()
    
    # Register the signal function handler
    signal.signal(signal.SIGALRM, handler)
    
    try:
        with contextlib.redirect_stdout(output_buffer):
            # Set the alarm
            signal.alarm(timeout)
            exec(code, {'__name__': '__main__'})
            # Disable the alarm
            signal.alarm(0)
        return output_buffer.getvalue().strip()
    except TimeoutError:
        return "Error: Execution timed out"
    except Exception as e:
        signal.alarm(0) # Ensure alarm is disabled on error
        return f"Error: {e}"

def predict(test: pl.DataFrame, sample_submission: pl.DataFrame):
    problem_text = test['problem'][0]
    problem_id = test['id'][0]
    
    print(f"Solving Problem {problem_id}...")
    
    if not API_KEY:
        print("CRITICAL: DEEPSEEK_API_KEY not set!")
        return sample_submission.with_columns(pl.lit(0).alias('answer'))

    # 1. Ask DeepSeek
    llm_response = solve_with_deepseek(problem_text, API_KEY)
    
    # 2. Extract Code
    code = extract_code(llm_response)
    
    answer = 0
    if code:
        print("Code generated. Executing...")
        # 3. Execute Code
        result = execute_python_code(code, timeout=7) # 7 seconds timeout
        print(f"Execution Result: {result}")
        
        # 4. Parse Answer (Look for BOXED_ANSWER)
        try:
            import re
            # Look for explicit format first
            match = re.search(r'BOXED_ANSWER:\s*(\d+)', result)
            if match:
                answer = int(match.group(1))
            else:
                # Fallback: Try to find the last integer
                numbers = re.findall(r'\d+', result)
                if numbers:
                    answer = int(numbers[-1])
            
            answer = answer % 100000 # Ensure within range
        except:
            pass
    else:
        print("No code found in response.")

    return sample_submission.with_columns(pl.lit(answer).alias('answer'))

def main():
    inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
    inference_server.serve()

if __name__ == "__main__":
    main()
