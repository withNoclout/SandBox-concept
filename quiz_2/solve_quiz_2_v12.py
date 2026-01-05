import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
print("Loading Data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Combine for Group Analysis
test['Survived'] = np.nan
all_data = pd.concat([train, test], sort=False)

# 2. Feature Engineering (Basic)
print("Basic Feature Engineering...")
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')

# Map Sex
all_data['Sex'] = all_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Fill Missing
all_data['Age'] = all_data['Age'].fillna(all_data.groupby('Title')['Age'].transform('median'))
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
all_data['Embarked'] = all_data['Embarked'].fillna('S')

# 3. The Detective: Ticket Grouping
print("Identifying Groups...")
# Extract Surname
all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0])

# Group by Ticket
all_data['Ticket_Freq'] = all_data.groupby('Ticket')['Ticket'].transform('count')

# Identify "Woman-Child" groups
all_data['IsWC'] = ((all_data['Sex'] == 1) | (all_data['Title'] == 'Master')).astype(int)

# 4. Train Baseline Model (Random Forest)
print("Training Baseline Model...")
features = ['Pclass', 'Sex', 'Age', 'Fare']
train_df = all_data[all_data['Survived'].notna()]
test_df = all_data[all_data['Survived'].isna()].copy()

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rf.fit(train_df[features], train_df['Survived'])
test_df['Survived'] = rf.predict(test_df[features])

# 5. Apply WCG Corrections
print("Applying WCG Corrections...")

# We need to look at the FATE of the group in the TRAIN set
# For a passenger in Test, look at their group members in Train.

# Create a lookup for Ticket Survival
# 1. Females/Children (Boy = Master)
# If a group has female/child survivors in Train, predict female/child in Test to Survive.
# If a group has female/child victims in Train, predict female/child in Test to Die.

# Identify "Woman-Child" groups
all_data['IsWC'] = ((all_data['Sex'] == 1) | (all_data['Title'] == 'Master')).astype(int)

# Calculate Group Survival Rate (excluding self)
# This is tricky across Train/Test.
# Simplified approach:
# For each Ticket, calculate 'WC_Survived' count and 'WC_Died' count in TRAIN.

ticket_stats = train_df.groupby('Ticket')['Survived'].agg(['count', 'sum', 'mean'])
# We specifically care about WC fate
wc_train = train_df[train_df['IsWC'] == 1]
wc_ticket_stats = wc_train.groupby('Ticket')['Survived'].agg(['count', 'sum', 'mean'])

# Identify "Dead Groups" (All WC died)
dead_tickets = wc_ticket_stats[wc_ticket_stats['mean'] == 0.0].index

# Identify "Alive Groups" (All WC lived)
alive_tickets = wc_ticket_stats[wc_ticket_stats['mean'] == 1.0].index

print(f"Found {len(dead_tickets)} Dead Tickets and {len(alive_tickets)} Alive Tickets.")

# Correction 1: Females/Masters in "Dead Tickets" -> DIE
# (Baseline likely predicts them to Live because they are Female)
mask_die = (test_df['Ticket'].isin(dead_tickets)) & (test_df['IsWC'] == 1)
print(f"Correcting {mask_die.sum()} passengers to DIE.")
test_df.loc[mask_die, 'Survived'] = 0

# Correction 2: Males in "Alive Tickets" -> LIVE?
# Usually WCG only applies to WC.
# But sometimes "All Lived" implies family survival.
# Let's stick to the classic WCG:
# - If Ticket has WC survivors -> Predict WC to Live (Baseline already does this usually)
# - If Ticket has WC victims -> Predict WC to Die (Baseline gets this WRONG)

# What about Males?
# If all males in a ticket LIVED, maybe predict Live?
male_train = train_df[train_df['Sex'] == 0]
male_ticket_stats = male_train.groupby('Ticket')['Survived'].agg(['count', 'sum', 'mean'])
male_alive_tickets = male_ticket_stats[male_ticket_stats['mean'] == 1.0].index

mask_live = (test_df['Ticket'].isin(male_alive_tickets)) & (test_df['Sex'] == 0)
print(f"Correcting {mask_live.sum()} males to LIVE.")
test_df.loc[mask_live, 'Survived'] = 1

# 6. Submission
print("Saving Submission...")
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_df['Survived'].astype(int)
})
submission.to_csv('submission_v12.csv', index=False)
print("Saved 'submission_v12.csv'")
