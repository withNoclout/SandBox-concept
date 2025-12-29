import os
import requests
import json
import re

# You will need to set this environment variable
# export DEEPSEEK_API_KEY="your_key_here"

API_URL = "https://api.deepseek.com/v1/chat/completions" # Verify exact URL

SYSTEM_PROMPT = """You are an expert mathematician and Python programmer participating in a math olympiad.
Your goal is to solve complex mathematical problems by writing and executing Python code.

**Instructions:**
1.  **Think First:** Briefly analyze the problem and outline a plan.
2.  **Use Python:** Write Python code to calculate the answer. DO NOT try to solve it mentally.
    *   Use `sympy` for symbolic math (algebra, calculus).
    *   Use `math` or `numpy` for numerical tasks.
    *   Implement simulations or brute-force searches if the problem space is small.
3.  **Verification:** In your code, try to verify the answer if possible (e.g., plug it back into the equation).
4.  **Final Output:**
    *   The code MUST print the final answer.
    *   The final answer must be a non-negative integer between 0 and 99999.
    *   Print the answer in this exact format: `print(f"BOXED_ANSWER: {answer}")`

**Example:**
Problem: What is 12 + 12?
Code:
```python
result = 12 + 12
print(f"BOXED_ANSWER: {result}")
```
"""

def solve_with_deepseek(problem_text, api_key):
    if not api_key:
        return "Error: No API Key provided"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-coder", # or deepseek-chat, check availability
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Solve this problem:\n{problem_text}"}
        ],
        "temperature": 0.0 # Deterministic for math
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        return content
    except Exception as e:
        return f"Error calling API: {e}"

def extract_code(llm_response):
    """Extracts Python code block from the LLM response."""
    code_match = re.search(r'```python(.*?)```', llm_response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    return None
