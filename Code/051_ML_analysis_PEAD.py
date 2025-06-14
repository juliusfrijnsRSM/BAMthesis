#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:05:54 2025

@author: juliusfrijns
"""

#%% ML analysis for PEAD part

# Imports
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform, uniform
from joblib import dump, load

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap

# Define random state and keep constant
RANDOM_STATE = 678329 # <- my student number

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

cols_to_drop = [
    "YearReport",
    "Country",
    "Industry",
    "SubIndustry"   
]
X_reduced = X.drop(columns=cols_to_drop)

# Original
X_train_original, X_test_original, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
    )

# Ordinal encoding
string_cols = X.select_dtypes(include=["object", "string"]).columns

if len(string_cols) > 0:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[string_cols] = encoder.fit_transform(X[string_cols])
    
    # Train/test split
    X_train_ordinal, X_test_ordinal, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
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
    X_onehot, y, test_size=0.2, random_state=RANDOM_STATE
    )
X_train_onehot_reduced, X_test_onehot_reduced, y_train, y_test = train_test_split(
    X_onehot_reduced, y, test_size=0.2, random_state=RANDOM_STATE
    )

# Label encoding
X_train_label = X_train_original.copy()
X_test_label = X_test_original.copy()
if len(string_cols) > 0:
    for col in string_cols:
        le = LabelEncoder()
        X_train_label[col] = le.fit_transform(X_train_original[col])
        X_test_label[col] = le.transform(X_test_original[col])
    
# Create empty dict to append best model to
best_models = dict()
best_models_params = dict()
best_models_stats = dict()

#%% Elastic net model

model_name = "ElasticNet"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot_reduced.copy()
X_test = X_test_onehot_reduced.copy()

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', ElasticNet(max_iter=5000, random_state=RANDOM_STATE))
])

# Grid search parameter grid
param_grid = {
    'reg__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],  # log scale manually
    'reg__l1_ratio': np.round(np.linspace(0.0, 1.0, 11), 2).tolist()  # Ridge to Lasso
}

# Grid search
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}


#%% KNN model

model_name = "KNN"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot_reduced.copy()
X_test = X_test_onehot_reduced.copy()

# Param grid
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev'],
    'p': [1, 2]  # Only relevant if metric='minkowski'
}


# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model
knn = KNeighborsRegressor()

# Grid search
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # or 'r2'
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train_scaled, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}


#%% Random forest model

model_name = "Random Forest"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_ordinal.copy()
X_test = X_test_ordinal.copy()

# Tuning grid
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}


# Define model
model = RandomForestRegressor(random_state=RANDOM_STATE)

# Grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # or 'r2'
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}


#%% XGBoost model

model_name = "XGBoost"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_ordinal.copy()
X_test = X_test_ordinal.copy()

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
model = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Set up grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # or 'r2', or 'neg_root_mean_squared_error'
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}



#%% AdaBoost model

model_name = "AdaBoost"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_ordinal.copy()
X_test = X_test_ordinal.copy()

# Define base estimators
base_estimators = [
    DecisionTreeRegressor(max_depth=i) for i in range(1, 4)
]

# Define tuning grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'estimator': base_estimators
}

# Define model
model = AdaBoostRegressor(random_state=RANDOM_STATE)

# Define grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # or 'r2' or 'neg_root_mean_squared_error'
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}


#%% MLP model

model_name = "MLP"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot_reduced.copy()
X_test = X_test_onehot_reduced.copy()

# Tuning grid
param_grid = {
    'hidden_layer_sizes': [
        (32,), (64,), (128,), (256,),
        (64, 32), (128, 64), (256, 128)
    ],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1e-5, 1e-4, 1e-3],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.0005, 0.001, 0.005, 0.01],
    'max_iter': [1000],
}


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model
mlp = MLPRegressor(random_state=RANDOM_STATE)

# Grid search
grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # or 'r2', or 'neg_root_mean_squared_error'
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit model
grid_search.fit(X_train_scaled, y_train)

# Predict and evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}


#%% LightGBM model

model_name = "LightGBM"
print(f"\n{model_name}")

# Use the same features
X_train = X_train_label.copy()
X_test = X_test_label.copy()

# Define parameter grid for regression
param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

# Initialize regressor
lgbm_reg = LGBMRegressor(random_state=RANDOM_STATE)

# Grid search
grid_search = GridSearchCV(
    estimator=lgbm_reg,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # or use your own scorer
    cv=5,
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Refit best model
best_model = LGBMRegressor(
    **grid_search.best_params_,
    random_state=RANDOM_STATE
)
best_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}


#%% CatBoost model

model_name = "CatBoost"
print(f"\n{model_name}")

# Use the same features
X_train = X_train_original.copy()
X_test = X_test_original.copy()

# Get and convert categorical columns
cat_features = [X_train.columns.get_loc(col) for col in string_cols]

for col in string_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Define hyperparameter grid
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    'iterations': [200],
    'l2_leaf_reg': [1, 3, 5]
}

# Define base model
catboost_reg = CatBoostRegressor(
    verbose=0,
    random_state=RANDOM_STATE,
    loss_function='RMSE'
)

# Prepare fit_params to pass cat_features
fit_params = {
    'cat_features': cat_features
}

# Grid search
grid_search = GridSearchCV(
    estimator=catboost_reg,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # or custom scorer
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train, **fit_params)

# Refit best model
best_model = CatBoostRegressor(
    **grid_search.best_params_,
    verbose=0,
    random_state=RANDOM_STATE,
    loss_function='RMSE'
)
best_model.fit(X_train, y_train, cat_features=cat_features)

# Predict and evaluate
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}



#%% Print final scores
print("\n")
print("------------------- Final model stats ---------------------")

for mod in best_models_stats.keys():
    print(f"\n{mod}")
    
    print(f"MSE: {best_models_stats[mod]['mse']}")
    print(f"RMSE: {best_models_stats[mod]['rmse']}")
    print(f"MAE: {best_models_stats[mod]['mae']}")
    print(f"R2: {best_models_stats[mod]['r2']}")
    

# Convert to DataFrame
df_stats = pd.DataFrame(best_models_stats).T.reset_index()
df_stats.columns = ['Model', 'MSE', 'RMSE', 'MAE', 'R2']

# Sort by RMSE (lower is better)
df_stats = df_stats.sort_values(by='RMSE')
df_stats.index = df_stats["Model"]
df_stats = df_stats.drop(columns=["Model"])

# Print
print(df_stats)
latex_ready = df_stats.map(lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, float) else x)
print(latex_ready.to_latex())



#%% Variable importance

best_mod_str = df_stats.index[0]
best_model = best_models[best_mod_str]

# Save best model externally to prevent having to tune and train all the time
dump(best_model, "Models/best_mod_regression.joblib")


booster = best_model.get_booster()
importance_dict = booster.get_score(importance_type='gain')

# Convert to sorted DataFrame
importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', columns=['Importance'])
importance_df.index.name = 'Feature'
importance_df = importance_df.reset_index().sort_values(by='Importance', ascending=False).head(15)

# Plot
plt.figure(figsize=(8,6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title("Top 15 Most Important Features (XGBoost)")
plt.tight_layout()
plt.savefig("Plots/feature_importance/feature_importance_drift.png")
plt.show()


# Create SHAP explainer and compute SHAP values
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test_ordinal)

# Plot SHAP summary (inline)
shap.summary_plot(shap_values, X_test_ordinal, plot_type="dot")

# Save the same plot
shap.summary_plot(shap_values, X_test_ordinal, plot_type="dot", show=False)
fig = plt.gcf()
fig.tight_layout()
fig.savefig("Plots/feature_importance/feature_importance_shap_regression.png", dpi=300, bbox_inches="tight")
plt.close(fig)