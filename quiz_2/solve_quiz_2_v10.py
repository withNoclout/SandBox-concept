import pandas as pd
import os

# 1. Load Submissions
print("Loading submissions...")

# Check if files exist
if not os.path.exists('submission_v8.csv'):
    print("Error: submission_v8.csv not found! Please run solve_quiz_2_v8.py first.")
    exit()

if not os.path.exists('submission_v6.csv'):
    print("Error: submission_v6.csv not found! Please run solve_quiz_2_v6.py first.")
    exit()

v8 = pd.read_csv('submission_v8.csv')
v6 = pd.read_csv('submission_v6.csv')

print(f"V8 Shape: {v8.shape}")
print(f"V6 Shape: {v6.shape}")

# 2. Blend
# V8 (0.78468) is our Champion. It gets high weight.
# V6 (0.77272) is our Complex Challenger. It gets low weight to provide diversity.
w_v8 = 0.7
w_v6 = 0.3

print(f"Blending: {w_v8} * V8 + {w_v6} * V6")

# Note: Since predictions are 0/1, we average them and then threshold.
# This works because the CSVs contain 0/1 integers.
# If we had probabilities, averaging probabilities would be better, but we only have the hard predictions saved.
# Averaging hard predictions is effectively a "Weighted Vote".
# If V8 says 1 and V6 says 0 -> 0.7. Threshold 0.5 -> 1. (V8 wins)
# If V8 says 0 and V6 says 1 -> 0.3. Threshold 0.5 -> 0. (V8 wins)
# Wait, if we just average hard predictions with 0.7/0.3, V8 will ALWAYS win disagreements.
# That's pointless.
# We need to re-generate PROBABILITIES if we want a true blend.
# BUT, we don't want to re-run the heavy training of V6.
# Let's check if we saved probabilities... we didn't. We saved hard classes.

# ALTERNATIVE STRATEGY:
# Since we only have hard classes, we can't do a weighted average that respects the minority unless the weights are closer (e.g. 0.4/0.6).
# But V8 is much better.
# Actually, let's look at the files.
# If we want to blend, we should probably re-run the models to get probabilities.
# However, for V10, let's try a different approach:
# We will trust V8, but if V6 is VERY confident... wait we don't have confidence.

# Let's re-implement a quick blend by re-loading the models? 
# No, V6 takes forever (Optuna).
# V8 is fast.

# Let's just do a 50/50 blend?
# If they disagree, who wins?
# If we do 0.5 * V8 + 0.5 * V6 -> 0.5. Threshold 0.5 -> 1? Or 0?
# Usually >= 0.5 is 1. So if EITHER is 1, it becomes 1? That's an "OR" gate.
# That increases survival rate.

# Let's stick to the plan but maybe adjust weights or acknowledge the limitation.
# Actually, let's try to be smarter.
# We can load the V6 script and run it with the BEST params found (hardcoded) to get probabilities!
# That avoids the Optuna search.
# Yes!

print("Re-generating probabilities for V6 (using best params)...")
# We need to import the code from V6 but just run the final part.
# Instead of importing, I'll just copy the essential parts here for a standalone script.
# It's safer and cleaner.

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- DATA LOADING & PREP (Combined) ---
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
passenger_ids = test_df['PassengerId']

# --- V8 PREP & MODEL ---
def get_v8_probs():
    # Minimal Features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = train_df[features].copy()
    y = train_df['Survived']
    X_test = test_df[features].copy()

    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

    lr = LogisticRegression(random_state=1, solver='liblinear')
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
    model = Pipeline(steps=[('preprocessor', preprocessor), ('ensemble', ensemble)])
    
    model.fit(X, y)
    return model.predict_proba(X_test)[:, 1]

# --- V6 PREP & MODEL ---
def get_v6_probs():
    # Oblique Features
    def process_data(df):
        df = df.copy()
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df['Title'] = df['Title'].map(title_mapping).fillna(0)
        df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
        df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'M')
        deck_mapping = {"A": 1, "B": 1, "C": 1, "D": 2, "E": 2, "F": 3, "G": 3, "T": 3, "M": 4}
        df['Deck'] = df['Deck'].map(deck_mapping).fillna(4).astype(int)
        
        # Oblique
        df['Age_Class'] = df['Age'] * df['Pclass']
        df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
        df['Age_Sex'] = df['Age'] * df['Sex']
        df['Class_Sex'] = df['Pclass'] * df['Sex']
        df['Fare_Age'] = df['Fare'] * df['Age']
        
        df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
        return df

    train_proc = process_data(train_df)
    test_proc = process_data(test_df)
    X = train_proc.drop("Survived", axis=1)
    y = train_proc["Survived"]
    X_test = test_proc.copy()

    # Best Params from V6 (Hardcoded)
    best_params = {'n_estimators': 875, 'max_depth': 7, 'learning_rate': 0.012412866805251027, 'subsample': 0.9851753324361814, 'colsample_bytree': 0.7391753595045609, 'gamma': 0.2827671434252517, 'reg_alpha': 4.043798930416074, 'reg_lambda': 3.648361758061688, 'min_child_weight': 8, 'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 1, 'n_jobs': 4}
    
    model = XGBClassifier(**best_params)
    model.fit(X, y)
    return model.predict_proba(X_test)[:, 1]

print("Generating V8 Probabilities...")
probs_v8 = get_v8_probs()

print("Generating V6 Probabilities...")
probs_v6 = get_v6_probs()

# BLEND
print(f"Blending Probabilities: {w_v8} * V8 + {w_v6} * V6")
final_probs = (w_v8 * probs_v8) + (w_v6 * probs_v6)
final_preds = (final_probs >= 0.5).astype(int)

# 3. Save
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_preds})
output.to_csv('submission_v10.csv', index=False)
print("Saved 'submission_v10.csv'")

with open('prediction_summary_v10.txt', 'w') as f:
    f.write("=== Titanic V10 (The Grand Blend) ===\n")
    f.write("Strategy: Weighted Average of Probabilities\n")
    f.write(f"Weights: {w_v8} * V8 (Minimalist) + {w_v6} * V6 (Imitator)\n")
    f.write("Rationale: Combining the stability of Linear/RF with the complexity of Oblique XGBoost.\n")

print("Saved 'prediction_summary_v10.txt'")
