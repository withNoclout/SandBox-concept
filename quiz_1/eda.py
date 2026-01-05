import pandas as pd

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("=== Train Data Shape ===")
print(train_df.shape)
print("\n=== Test Data Shape ===")
print(test_df.shape)

print("\n=== Columns with Missing Values (Train) ===")
missing = train_df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

print("\n=== Target Variable (SalePrice) Stats ===")
print(train_df['SalePrice'].describe())

print("\n=== Numerical Features Correlation with SalePrice (Top 10) ===")
numeric_features = train_df.select_dtypes(include=['int64', 'float64'])
correlation = numeric_features.corr()['SalePrice'].sort_values(ascending=False)
print(correlation.head(11)) # Top 10 + SalePrice itself
