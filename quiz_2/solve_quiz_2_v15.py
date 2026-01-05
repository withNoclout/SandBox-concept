import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
print("Loading Data...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 2. Prepare Data (Exactly as in the snippet)
print("Preparing Data...")
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# 3. Train Model
print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# 4. Predict
print("Predicting...")
predictions = model.predict(X_test)

# 5. Save Submission
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission_v15.csv', index=False)
print("Your submission was successfully saved as 'submission_v15.csv'!")
