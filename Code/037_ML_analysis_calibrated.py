#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:02:24 2025

@author: juliusfrijns
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:10:05 2025

@author: juliusfrijns
"""


#%% ML part of thesis

# Imports
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
#from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt

# Define random state and keep constant
RANDOM_STATE = 678329 # <- my student number

# Define threshold grid
threshold_grid = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Scorer function
def f1_movement_only(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[0, 2], average='macro', zero_division=0)

custom_scorer = make_scorer(f1_movement_only)

# Import data blocks/batches
with open("Block data/batches_dtypes.json", "r") as f:
    dtypes = json.load(f)
df = pd.read_csv("Block data/batches_prepared.csv", dtype=dtypes)
df = df.iloc[:, 1:]

#%% Data preparation

# Delete NaNs
#df_na = df.copy()
df = df.dropna()

# Delete infinite rows
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[~np.isinf(df[numeric_cols]).any(axis=1)]

# Split into features and target
X = df.iloc[:, 2:-4] # Take out Ticker and Company to prevent leakage/overfitting
y = df.iloc[:, -1]

cols_to_drop = [
    "YearReport",
    "Country",
    "Industry",
    "SubIndustry"   
]
X_reduced = X.drop(columns=cols_to_drop)

# Ordinal encoding
string_cols = X.select_dtypes(include=["object", "string"]).columns

if len(string_cols) > 0:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[string_cols] = encoder.fit_transform(X[string_cols])
    
    # Train/test split
    X_train_ordinal, X_test_ordinal, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,stratify=y
        )

# Onehot encoding
string_cols_reduced = X_reduced.select_dtypes(include=["object", "string"]).columns

if len(string_cols) > 0:
    
    X_onehot = pd.get_dummies(
        X, columns=string_cols, dummy_na=False, drop_first=True
        )
    
    X_onehot_reduced = pd.get_dummies(
        X_reduced, columns=string_cols_reduced, dummy_na=False, drop_first=True
        )

# Train/test split
X_train_onehot, X_test_onehot, y_train, y_test = train_test_split(
    X_onehot, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
X_train_onehot_reduced, X_test_onehot_reduced, y_train, y_test = train_test_split(
    X_onehot_reduced, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

# Remapping for multiclass modelling
y_train = y_train + 1
y_test = y_test + 1
    
# Create empty dict to append best model to
best_models = dict()
best_models_params = dict()
best_models_stats = dict()
best_models_thresholds = dict()
best_models_probas = dict()

#%% Logistic model

model_name = "Logistic"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot_reduced
X_test = X_test_onehot_reduced

# Pipeline
pipe = Pipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        solver='saga',
        class_weight='balanced',
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

# Grid search parameters
param_grid = [
    {'clf__penalty': ['l1'], 'clf__C': [0.01, 0.1, 1.0]},
    {'clf__penalty': ['l2'], 'clf__C': [0.01, 0.1, 1.0]},
    {'clf__penalty': ['elasticnet'], 'clf__C': [0.01, 0.1, 1.0], 'clf__l1_ratio': [0.0, 0.5, 1.0]}
]


# Grid search
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_

# Predict probabilities
proba = best_model.predict_proba(X_test)

# Threshold tuning loop
best_thresh = None
best_f1 = -1
best_y_pred = None


for t in threshold_grid:
    y_pred_custom = np.ones(len(proba), dtype=int)
    y_pred_custom[proba[:, 0] > t] = 0
    y_pred_custom[(proba[:, 2] > t) & (proba[:, 0] <= t)] = 2

    # Class 0
    prec0 = precision_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    rec0 = recall_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    f1_0 = 2 * (prec0 * rec0) / (prec0 + rec0 + 1e-10)

    # Class 2
    prec2 = precision_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    rec2 = recall_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    f1_2 = 2 * (prec2 * rec2) / (prec2 + rec2 + 1e-10)

    f1 = (f1_0 + f1_2) / 2

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
        best_y_pred = y_pred_custom

# Final evaluation
print("Final results with custom threshold")
print("Best Parameters:", grid_search.best_params_)
print(f"Best threshold for class 0/2 F1: {best_thresh}")
print("F1 (class 0 & 2 only):", best_f1)
print(classification_report(y_test, best_y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(
    y_test, best_y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True
)
best_models_thresholds[model_name] = best_thresh
best_models_probas[model_name] = proba


#%% KNN model

model_name = "KNN"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot_reduced
X_test = X_test_onehot_reduced

# Tuning grid
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['minkowski'],
    'knn__p': [1, 2]
}

# Pipeline
pipe = Pipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Grid search
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_

# Predict probabilities
proba = best_model.predict_proba(X_test)

# Threshold tuning loop
best_thresh = None
best_f1 = -1
best_y_pred = None


for t in threshold_grid:
    y_pred_custom = np.ones(len(proba), dtype=int)
    y_pred_custom[proba[:, 0] > t] = 0
    y_pred_custom[(proba[:, 2] > t) & (proba[:, 0] <= t)] = 2

    # Class 0
    prec0 = precision_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    rec0 = recall_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    f1_0 = 2 * (prec0 * rec0) / (prec0 + rec0 + 1e-10)

    # Class 2
    prec2 = precision_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    rec2 = recall_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    f1_2 = 2 * (prec2 * rec2) / (prec2 + rec2 + 1e-10)

    f1 = (f1_0 + f1_2) / 2

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
        best_y_pred = y_pred_custom

# Final evaluation
print("Final results with custom threshold")
print("Best Parameters:", grid_search.best_params_)
print(f"Best threshold for class 0/2 F1: {best_thresh}")
print("F1 (class 0 & 2 only):", best_f1)
print(classification_report(y_test, best_y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(
    y_test, best_y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True
)
best_models_thresholds[model_name] = best_thresh
best_models_probas[model_name] = proba


#%% Random forest model

model_name = "Random Forest"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_ordinal
X_test = X_test_ordinal

# Tuning grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

# Define model and grid search
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = CalibratedClassifierCV(estimator=grid_search.best_estimator_, method='sigmoid', cv='prefit')
best_model.fit(X_train, y_train)

# Predict probabilities
proba = best_model.predict_proba(X_test)

# Threshold tuning loop
best_thresh = None
best_f1 = -1
best_y_pred = None


for t in threshold_grid:
    y_pred_custom = np.ones(len(proba), dtype=int)
    y_pred_custom[proba[:, 0] > t] = 0
    y_pred_custom[(proba[:, 2] > t) & (proba[:, 0] <= t)] = 2

    # Class 0
    prec0 = precision_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    rec0 = recall_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    f1_0 = 2 * (prec0 * rec0) / (prec0 + rec0 + 1e-10)

    # Class 2
    prec2 = precision_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    rec2 = recall_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    f1_2 = 2 * (prec2 * rec2) / (prec2 + rec2 + 1e-10)

    f1 = (f1_0 + f1_2) / 2

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
        best_y_pred = y_pred_custom

# Final evaluation
print("Final results with custom threshold")
print("Best Parameters:", grid_search.best_params_)
print(f"Best threshold for class 0/2 F1: {best_thresh}")
print("F1 (class 0 & 2 only):", best_f1)
print(classification_report(y_test, best_y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(
    y_test, best_y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True
)
best_models_thresholds[model_name] = best_thresh
best_models_probas[model_name] = proba


#%% XGBoosting model

model_name = "XGBoost"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_ordinal
X_test = X_test_ordinal

# Set tuning grid
param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6],
    'min_child_weight': [1, 3],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0, 1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 5]
}

# Define model
model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,               # since you have classes 0, 1, 2
    eval_metric='mlogloss',    # avoid deprecated defaults
    random_state=RANDOM_STATE,
    n_jobs=-1                  # full parallel processing
)

# Set up grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_

# Predict probabilities
proba = best_model.predict_proba(X_test)

# Threshold tuning loop
best_thresh = None
best_f1 = -1
best_y_pred = None


for t in threshold_grid:
    y_pred_custom = np.ones(len(proba), dtype=int)
    y_pred_custom[proba[:, 0] > t] = 0
    y_pred_custom[(proba[:, 2] > t) & (proba[:, 0] <= t)] = 2

    # Class 0
    prec0 = precision_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    rec0 = recall_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    f1_0 = 2 * (prec0 * rec0) / (prec0 + rec0 + 1e-10)

    # Class 2
    prec2 = precision_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    rec2 = recall_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    f1_2 = 2 * (prec2 * rec2) / (prec2 + rec2 + 1e-10)

    f1 = (f1_0 + f1_2) / 2

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
        best_y_pred = y_pred_custom

# Final evaluation
print("Final results with custom threshold")
print("Best Parameters:", grid_search.best_params_)
print(f"Best threshold for class 0/2 F1: {best_thresh}")
print("F1 (class 0 & 2 only):", best_f1)
print(classification_report(y_test, best_y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(
    y_test, best_y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True
)
best_models_thresholds[model_name] = best_thresh
best_models_probas[model_name] = proba


#%% AdaBoost model

model_name = "AdaBoost"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_ordinal
X_test = X_test_ordinal

# Define base estimators
base_estimators = [
    DecisionTreeClassifier(max_depth=i, class_weight='balanced') for i in range(1, 4)
    ]

# Define tuning grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'estimator': base_estimators
}

# Define model
model = AdaBoostClassifier(random_state=RANDOM_STATE)

# Define grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = CalibratedClassifierCV(estimator=grid_search.best_estimator_, method='sigmoid', cv='prefit')
best_model.fit(X_train, y_train)

# Predict probabilities
proba = best_model.predict_proba(X_test)

# Threshold tuning loop
best_thresh = None
best_f1 = -1
best_y_pred = None


for t in threshold_grid:
    y_pred_custom = np.ones(len(proba), dtype=int)
    y_pred_custom[proba[:, 0] > t] = 0
    y_pred_custom[(proba[:, 2] > t) & (proba[:, 0] <= t)] = 2

    # Class 0
    prec0 = precision_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    rec0 = recall_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    f1_0 = 2 * (prec0 * rec0) / (prec0 + rec0 + 1e-10)

    # Class 2
    prec2 = precision_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    rec2 = recall_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    f1_2 = 2 * (prec2 * rec2) / (prec2 + rec2 + 1e-10)

    f1 = (f1_0 + f1_2) / 2

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
        best_y_pred = y_pred_custom

# Final evaluation
print("Final results with custom threshold")
print("Best Parameters:", grid_search.best_params_)
print(f"Best threshold for class 0/2 F1: {best_thresh}")
print("F1 (class 0 & 2 only):", best_f1)
print(classification_report(y_test, best_y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(
    y_test, best_y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True
)
best_models_thresholds[model_name] = best_thresh
best_models_probas[model_name] = proba


#%% MLP model

model_name = "MLP"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot_reduced
X_test = X_test_onehot_reduced

# Set tuning grid
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 128)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [1000],
}

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply weights to samples to counter class imbalance
sample_weights = compute_sample_weight(class_weight={0: 3, 1: 1, 2: 3}, y=y_train)


# Define model
mlp = MLPClassifier(early_stopping=True, random_state=RANDOM_STATE)

# Set up grid search
grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = CalibratedClassifierCV(estimator=grid_search.best_estimator_, method='sigmoid', cv='prefit')
best_model.fit(X_train, y_train)

# Predict probabilities
proba = best_model.predict_proba(X_test)

# Threshold tuning loop
best_thresh = None
best_f1 = -1
best_y_pred = None


for t in threshold_grid:
    y_pred_custom = np.ones(len(proba), dtype=int)
    y_pred_custom[proba[:, 0] > t] = 0
    y_pred_custom[(proba[:, 2] > t) & (proba[:, 0] <= t)] = 2

    # Class 0
    prec0 = precision_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    rec0 = recall_score(y_test, y_pred_custom, labels=[0], average='macro', zero_division=0)
    f1_0 = 2 * (prec0 * rec0) / (prec0 + rec0 + 1e-10)

    # Class 2
    prec2 = precision_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    rec2 = recall_score(y_test, y_pred_custom, labels=[2], average='macro', zero_division=0)
    f1_2 = 2 * (prec2 * rec2) / (prec2 + rec2 + 1e-10)

    f1 = (f1_0 + f1_2) / 2

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
        best_y_pred = y_pred_custom

# Final evaluation
print("Final results with custom threshold")
print("Best Parameters:", grid_search.best_params_)
print(f"Best threshold for class 0/2 F1: {best_thresh}")
print("F1 (class 0 & 2 only):", best_f1)
print(classification_report(y_test, best_y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(
    y_test, best_y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True
)
best_models_thresholds[model_name] = best_thresh
best_models_probas[model_name] = proba


#%% Print the final stats

print("\n")
print("------------------- Final model stats ---------------------")

for mod in best_models_stats.keys():
    print(f"\n{mod}")
    print(best_models_stats[mod])
    

# Make final dataframe with all model stats
rows = []
for model_name, stats in best_models_stats.items():
    macro = stats['macro avg']
    weighted = stats['weighted avg']
    accuracy = stats['accuracy']
    row = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Macro F1': macro['f1-score'],
        'Macro Precision': macro['precision'],
        'Macro Recall': macro['recall'],
        'Weighted F1': weighted['f1-score'],
        'Weighted Precision': weighted['precision'],
        'Weighted Recall': weighted['recall'],
        'F1 (0)': stats['0']['f1-score'],
        'Precision (0)': stats['0']['precision'],
        'Recall (0)': stats['0']['recall'],
        'F1 (1)': stats['1']['f1-score'],
        'Precision (1)': stats['1']['precision'],
        'Recall (1)': stats['1']['recall'],
        'F1 (2)': stats['2']['f1-score'],
        'Precision (2)': stats['2']['precision'],
        'Recall (2)': stats['2']['recall'],
    }
    rows.append(row)

df_stats = pd.DataFrame(rows)

# Sort based on weighted F1
df_stats = df_stats.sort_values(by='Weighted F1', ascending=False)

print(df_stats)
print(df_stats.to_latex())
