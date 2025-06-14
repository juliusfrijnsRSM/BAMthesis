#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 16:20:37 2025

@author: juliusfrijns
"""

#%% ML analysis for PEAD part

# Imports
import json
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

# Define random state and keep constant
RANDOM_STATE = 678329 # <- my student number

iterations = 300
iterations_capped = 50

# Import data blocks/batches
with open("Block data/batches_postearn_dtypes.json", "r") as f:
    dtypes = json.load(f)
df = pd.read_csv("Block data/batches_postearn.csv", dtype=dtypes)
df = df.iloc[:, 1:]

#%% Dataprep for ML regression analysis

# Drop nans and non-surprise rows
#df_preserve = df.copy()
df = df[df["Surprise"] == 1].dropna()

# Split into features and target
X = df.iloc[:, 2:-10].drop(columns=["Surprise"])

# Excess return as y
y_1week = df["ExcessDrift1Week"]
y_1month = df["ExcessDrift1Month"]
y_2months = df["ExcessDrift2Months"]
y_3months = df["ExcessDrift3Months"]

##### Set y-variable
y = y_3months

# Encode string columns using OrdinalEncoder
string_cols = X.select_dtypes(include=["object", "string"]).columns
if len(string_cols) > 0:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[string_cols] = encoder.fit_transform(X[string_cols])
    
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
# Create empty dict to append best model to
best_models = dict()
best_models_params = dict()
best_models_stats = dict()

#%% Elastic net model

model_name = "ElasticNet"
print("\n", model_name)

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', ElasticNet(max_iter=5000, random_state=RANDOM_STATE))
])

# Very wide randomized search space
param_distributions = {
    'reg__alpha': loguniform(1e-5, 10),            # log scale to cover a wide range
    'reg__l1_ratio': uniform(0.0, 1.0)              # uniform between 0.0 and 1.0
}

# Grid search
random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=iterations,                         # Increase for more coverage
    scoring='neg_mean_squared_error',  # or 'r2'
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)
# Fit model
random_search.fit(X_train, y_train)

# Predict and evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print("\nBest Parameters:", random_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "r2": r2
}

#%% KNN model

model_name = "KNN"
print("\n", model_name)

# Param distributions (same as grid)
param_distributions = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev'],
    'p': [1, 2]  # Only used with Minkowski metric (not actually needed for euclidean/manhattan)
}

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model
knn = KNeighborsRegressor()

# Randomized search
random_search = RandomizedSearchCV(
    estimator=knn,
    param_distributions=param_distributions,
    n_iter=iterations,  # Choose based on compute time; 30 is a reasonable default
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)

# Fit model
random_search.fit(X_train_scaled, y_train)

# Predict and evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print("\nBest Parameters:", random_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "r2": r2
}


#%% Random forest model

model_name = "Random Forest"
print("\n", model_name)

# Tuning distributions (same as original grid)
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

# Define model
model = RandomForestRegressor(random_state=RANDOM_STATE)

# Randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=iterations_capped,  # replaces n_iter
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)

# Fit model
random_search.fit(X_train, y_train)

# Predict and evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print("\nBest Parameters:", random_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "r2": r2
}



#%% XGBoost model

model_name = "XGBoost"
print("\n", model_name)

# Parameter distributions
param_distributions = {
    'n_estimators': [100, 200, 300, 500],          # discrete is fine here
    'learning_rate': loguniform(1e-4, 0.3),
    'max_depth': [3, 5, 7, 10],                    # small integer list is OK
    'min_child_weight': [1, 3, 5],                 # usually discrete
    'subsample': uniform(0.5, 0.5),                # samples from 0.5 to 1.0
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': loguniform(1e-3, 10),
    'reg_alpha': loguniform(1e-4, 10),
    'reg_lambda': loguniform(1e-4, 10)
}

# Define model
model = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=iterations,  # Using externally defined variable
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)

# Fit model
random_search.fit(X_train, y_train)

# Predict and evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", random_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "r2": r2
}


#%% AdaBoost model

model_name = "AdaBoost"
print("\n", model_name)

# Define base estimators
base_estimators = [
    DecisionTreeRegressor(max_depth=i) for i in range(1, 4)
]

# Define parameter distributions for randomized search
param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': loguniform(1e-4, 0.3),
    'estimator': base_estimators
}

# Define model
model = AdaBoostRegressor(random_state=RANDOM_STATE)

# Randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=iterations_capped,  # uses your external variable
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Fit model
random_search.fit(X_train, y_train)

# Predict and evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", random_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "r2": r2
}


#%% MLP model

model_name = "MLP"
print("\n", model_name)

# Define parameter distributions
param_distributions = {
    'hidden_layer_sizes': [
        (32,), (64,), (128,), (256,),
        (64, 32), (128, 64), (256, 128)
    ],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': loguniform(1e-6, 1e-1),               # regularization strength (L2)
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': loguniform(1e-4, 1e-1),  # initial learning rate
    'max_iter': [1000]                             # keep fixed
}

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model
mlp = MLPRegressor(random_state=RANDOM_STATE)

# Randomized search
random_search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_distributions,
    n_iter=iterations,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Fit model
random_search.fit(X_train_scaled, y_train)

# Predict and evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", random_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "r2": r2
}


#%% Print final scores

print("\n")
print("------------------- Final model stats ---------------------")

for mod in best_models_stats.keys():
    print(f"\n{mod}")
    
    print(f"MSE: {best_models_stats[mod]['mse']}")
    print(f"RMSE: {best_models_stats[mod]['rmse']}")
    print(f"R2: {best_models_stats[mod]['r2']}")
    
    
# Convert to DataFrame
df_stats = pd.DataFrame(best_models_stats).T.reset_index()
df_stats.columns = ['Model', 'MSE', 'RMSE', 'R2']

# Sort by RMSE (lower is better)
df_stats = df_stats.sort_values(by='RMSE')

# Print
print(df_stats)

