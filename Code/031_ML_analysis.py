#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 21:55:31 2025

@author: juliusfrijns
"""

#%% ML part of thesis

# Imports
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
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
multiclass = True

# Delete NaNs
#df_na = df.copy()
df = df.dropna()

# Split into features and target
X = df.iloc[:, 2:-4]
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
best_models_stats = dict()

#%% Logistic model

model_name = "Logistic"

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        solver='saga',
        #multi_class='multinomial',
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
    scoring='f1_macro',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0)

#%% KNN model

model_name = "KNN"

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
    'p': [1, 2]
}

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model
knn = KNeighborsClassifier()

# Grid search
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0)


#%% Random forest model

model_name = "Random Forest"

# Tuning grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'class_weight': ['balanced']
}

# Custom scorer
custom_scorer = make_scorer(f1_movement_only)

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

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0)


#%% XGBoosting model

model_name = "XGBoost"

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
custom_scorer = make_scorer(f1_movement_only)

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
    scoring='f1_macro',  # or your custom F1 scorer for movement classes
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0)


#%% Adaboost model

model_name = "AdaBoost"

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
    scoring='f1_macro',  # or your f1_movement_only scorer
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0)


#%% MLP model

model_name = "MLP"

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

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0)


#%% Voting ensemble model

voting_model = VotingClassifier(
    estimators=[
        ('logreg', best_models["Logistic"]),
        ('knn', best_models["KNN"]),
        ('rf', best_models["Random Forest"]),
        ('xgb', best_models["XGBoost"]),
        ('ab', best_models["AdaBoost"]),
        ('mlp', best_models["MLP"])
    ],
    voting='soft',
    n_jobs=-1
)

# Fit to scaled training data (assuming all models expect scaled X)
voting_model.fit(X_train_scaled, y_train)

# Predict & evaluate
y_pred = voting_model.predict(X_test_scaled)

print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))



