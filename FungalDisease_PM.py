import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# LOAD TRAINING DATA
train_data = pd.read_csv('DataTrain_HW3Problem3.csv')
train_data.replace('NA', np.nan, inplace=True)

# LOAD TEST DATA
test_data = pd.read_csv('DataTest_HW3Problem3.csv')
test_data.replace('NA', np.nan, inplace=True)

# ENSURE NUMERIC TYPES
for col in train_data.columns:
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
for col in test_data.columns:
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

# FEATURES AND TARGET
feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5']
X_train = train_data[feature_cols]
y_train = train_data['y'].astype(int)
X_test = test_data[feature_cols]

# HISTGRADIENTBOOSTING CAN HANDLE MISSIN VALUES (NaNs) NATIVELY 
model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=None,
    max_iter=300,
    l2_regularization=0.0,
    random_state=42
)
model.fit(X_train, y_train)

# PREDICT ON TEST DATA
y_pred = model.predict(X_test)

# OUTPUT TO FILE
with open('predictions_hw3p3.txt', 'w') as f:
    for pred in y_pred:
        f.write(f"{int(pred)}\n")