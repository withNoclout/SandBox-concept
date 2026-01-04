import pandas as pd

def analyze_file(filename):
    print(f"--- Analyzing {filename} ---")
    try:
        df = pd.read_csv(filename)
        print("Shape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nHead:\n", df.head())
        print("\nInfo:")
        print(df.info())
        print("\nMissing Values:\n", df.isnull().sum())
        print("\n" + "="*30 + "\n")
    except Exception as e:
        print(f"Error analyzing {filename}: {e}")

if __name__ == "__main__":
    analyze_file("train.csv")
    analyze_file("test.csv")
