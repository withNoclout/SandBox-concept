from deepseek_solver import solve_with_deepseek, extract_code
import os

# Use the key provided by the user
API_KEY = os.environ.get("DEEPSEEK_API_KEY")

print(f"Testing API Key: {API_KEY[:5]}...{API_KEY[-5:] if API_KEY else 'None'}")

problem = "What is 12 + 12? Write a Python script to print the answer."
print(f"\nSending Problem: {problem}")

response = solve_with_deepseek(problem, API_KEY)
print("\n--- Raw Response ---")
print(response)

code = extract_code(response)
print("\n--- Extracted Code ---")
print(code)

if code:
    print("\n--- Executing Code ---")
    try:
        exec(code)
        print("\n✅ Execution Successful")
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
else:
    print("\n❌ No code found in response")
