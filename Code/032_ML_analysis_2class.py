#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 17:46:18 2025

@author: juliusfrijns
"""

#%% ML part of thesis

# Imports
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

import seaborn as sns
import matplotlib.pyplot as plt

# Define random state and keep constant
RANDOM_STATE = 678329 # <- my student number

# Number of estimators
n_estimators = 100

# Scorer function
def f1_movement_only(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[0, 2], average='macro', zero_division=0)

# Import data blocks/batches
with open("Block data/batches_dtypes.json", "r") as f:
    dtypes = json.load(f)
df = pd.read_csv("Block data/batches_prepared.csv", dtype=dtypes)
df = df.iloc[:, 1:]

#%% Data preparation

# Multiclass?
multiclass = False

# Delete NaNs
df_na = df.copy()
df = df.dropna()

# Split into features and target
X = df.iloc[:, :-4]
if multiclass:
    y = df.iloc[:, -1]
else:
    y = df.iloc[:, -4]

# Encode string columns using OrdinalEncoder
string_cols = X.select_dtypes(include=["object", "string"]).columns
if len(string_cols) > 0:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[string_cols] = encoder.fit_transform(X[string_cols])
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Remapping for multiclass modelling
if multiclass:
    y_train = y_train + 1
    y_test = y_test + 1
    
# Create empty dict to append best model to
best_models = dict()
best_models_params = dict()

#%% Random forest model

# Tuning grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'class_weight': [
        'balanced'
        #{0: 3, 1: 1, 2: 3}
        #{0: 5, 1: 1, 2: 5}
    ]
}

# Custom scorer
#custom_scorer = make_scorer(f1_movement_only)

# Define model and grid search
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE
)

model = RandomForestClassifier(random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1_macro',  # or 'accuracy', or 'balanced_accuracy'
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit models according to grid
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Append
best_models["Random Forest"] = best_model
best_models_params["Random Forest"] = grid_search.best_params_

# Predict
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))


#%% Gradient boosting model

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

# Custom scorer
#custom_scorer = make_scorer(f1_movement_only)

# Define model
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='mlogloss',
    random_state=RANDOM_STATE,
    n_jobs=-1                  # full parallel processing
)

# Set up grid search

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='f1',  # or your custom F1 scorer for movement classes
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit models according to grid
grid_search.fit(X_train, y_train)

# Evaluate
best_model = grid_search.best_estimator_

# Append
best_models["Gradient Boost"] = best_model
best_models_params["Gradient Boost"] = grid_search.best_params_

# Predict
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))


#%% MLP model

# Set tuning grid
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1e-5, 1e-4, 1e-3],
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
mlp = MLPClassifier(random_state=RANDOM_STATE)

# Set up grid search
grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring='f1_macro',  # or your custom f1_movement_only scorer
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Use grid search to find best model
grid_search.fit(X_train_scaled, y_train)


# Evaluate
best_model = grid_search.best_estimator_
best_model.fit(X_train_scaled, y_train)

# Append
best_models["MLP"] = best_model
best_models_params["MLP"] = grid_search.best_params_

# Predict
y_pred = best_model.predict(X_test_scaled)
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))






