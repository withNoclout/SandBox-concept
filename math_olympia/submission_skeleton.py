import kaggle_evaluation.aimo_3_inference_server
import polars as pl

def predict(test: pl.DataFrame, sample_submission: pl.DataFrame):
    """
    This function is called for each test problem.
    test: DataFrame with columns ['id', 'problem']
    sample_submission: DataFrame with columns ['id', 'answer']
    """
    # 1. Get the problem text
    problem_text = test['problem'][0]
    problem_id = test['id'][0]
    
    print(f"Solving Problem ID: {problem_id}")
    print(f"Problem: {problem_text}")
    
    # 2. TODO: Implement TIR (Tool-Integrated Reasoning) here
    # - Call LLM to generate Python code
    # - Execute Python code
    # - Get answer
    
    # Placeholder answer (must be integer 0-99999)
    answer = 0 
    
    return sample_submission.with_columns(pl.lit(answer).alias('answer'))

def main():
    # This starts the inference server which communicates with the Gateway
    inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)
    inference_server.serve()

if __name__ == "__main__":
    main()
