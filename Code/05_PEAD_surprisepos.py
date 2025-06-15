#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 17:53:21 2025

@author: juliusfrijns
"""

#%% ML analysis for PEAD part

# Imports
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import loguniform, uniform, randint
from joblib import dump, load

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
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

# Set split year
split_year1 = 2022
split_year2 = 2024

# Import data blocks/batches
with open("Block data/batches_postearn_FF5pre_dtypes.json", "r") as f:
    dtypes = json.load(f)
df = pd.read_csv("Block data/batches_postearn_FF5pre.csv", dtype=dtypes)
df = df.iloc[:, 1:]

#%% Dataprep for ML regression analysis

# Drop nans and non-surprise rows
df = df[df["SurprisePos"] == 1].dropna()

# Delete infinite rows
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[~np.isinf(df[numeric_cols]).any(axis=1)]

# Columns to drop for reduced dimensionality X sets
cols_to_drop = [
    #"YearReport",
    "Country",
    "Industry",
    "SubIndustry"   
]

# Define train and test dataframes based on year to split on
df_train = df[df["YearReport"] < split_year1]
df_test = df[(df["YearReport"] >= split_year1) & (df["YearReport"] < split_year2)]

df_train_test = df[df["YearReport"] < split_year2]
df_simulation = df[df["YearReport"] >= split_year2]


# Drop simulation column as it should not be used in model training
sim_cols_to_drop = [
    "AbnormalReturnSimulation", "RawReturnSimulation", 
    "PostAbnormalReturnSimulation",
    "MarketReturnSimulation", "PostMarketReturnSimulation"
    ]

df_train = df_train.drop(columns = sim_cols_to_drop)
df_test = df_test.drop(columns = sim_cols_to_drop)
df_train_test = df_train_test.drop(columns = sim_cols_to_drop)


################## Set y-variable #####################################3
y_train = df_train["PostAbnormalReturn2Months"]
y_test = df_test["PostAbnormalReturn2Months"]


# Original sets (no encoding or dimensionality reducing)
X_train_original = df_train.iloc[:, 2:-8]#.drop(columns="Surprise")
X_test_original = df_test.iloc[:, 2:-8]#.drop(columns="Surprise")


# Reduced sets
X_train_reduced = X_train_original.drop(columns=cols_to_drop)
X_test_reduced = X_test_original.drop(columns=cols_to_drop)


# Ordinal encoded sets
string_cols = X_train_original.select_dtypes(include=["object", "string"]).columns

X_train_ordinal = X_train_original.copy()
X_test_ordinal = X_test_original.copy()

if len(string_cols) > 0:
    
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    X_train_ordinal[string_cols] = encoder.fit_transform(X_train_ordinal[string_cols])
    X_test_ordinal[string_cols] = encoder.transform(X_test_ordinal[string_cols])


# Onehot encoded sets
string_cols_reduced = X_train_reduced.select_dtypes(include=["object", "string"]).columns

if len(string_cols) > 0:
    
    X_train_onehot = pd.get_dummies(
        X_train_original, columns=string_cols, dummy_na=False, drop_first=True
        )
    X_test_onehot = pd.get_dummies(
        X_test_original, columns=string_cols, dummy_na=False, drop_first=True
        )
    
    X_train_onehot_reduced = pd.get_dummies(
        X_train_reduced, columns=string_cols_reduced, dummy_na=False, drop_first=True
        )
    X_test_onehot_reduced = pd.get_dummies(
        X_test_reduced, columns=string_cols_reduced, dummy_na=False, drop_first=True
        )
    
    
# Label encoded sets
X_train_label = X_train_original.copy()
X_test_label = X_test_original.copy()

if len(string_cols) > 0:
    
    for col in string_cols:
    
        le = LabelEncoder()
        X_train_label[col] = le.fit_transform(X_train_original[col])
        X_test_label[col] = le.transform(X_test_original[col])
        

# Remove Year Report from all train and test sets
for tt_set in [
    X_train_original, X_test_original,
    X_train_reduced, X_test_reduced,
    X_train_ordinal, X_test_ordinal,
    X_train_label, X_test_label,
    X_train_onehot, X_test_onehot,
    X_train_onehot_reduced, X_test_onehot_reduced
]:
    if "YearReport" in tt_set.columns:
        tt_set.drop(columns="YearReport", inplace=True)



# Create empty dict to append best model to
best_models = dict()
best_models_params = dict()
best_models_stats = dict()


#%% OLS model

model_name = "OLS"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot.copy()
X_test = X_test_onehot.copy()

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
#print("Best Parameters:", grid_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = model
#best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "r2": r2
}


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
    'reg__alpha': loguniform(1e-4, 10),
    'reg__l1_ratio': uniform(0, 1)
}

# Randomized search
rand_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_grid,  # Reuse your param_grid as param_distributions
    n_iter=30,  # You can adjust this based on computational budget
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit model
rand_search.fit(X_train, y_train)

# Predict and evaluate
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", rand_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = rand_search.best_params_
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
    'n_neighbors': randint(1, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}


# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Randomized search for KNN
rand_search = RandomizedSearchCV(
    estimator=KNeighborsRegressor(),
    param_distributions={
        'n_neighbors': randint(1, 30),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev'],
        'p': [1, 2]  # Only relevant if metric='minkowski'
    },
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit model
rand_search.fit(X_train_scaled, y_train)

# Predict and evaluate
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", rand_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = rand_search.best_params_
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
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['auto', 'sqrt', 'log2']
}

# Randomized search for Random Forest
rand_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=RANDOM_STATE),
    param_distributions={
        'n_estimators': randint(100, 400),
        'max_depth': randint(10, 30),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit model
rand_search.fit(X_train, y_train)

# Predict and evaluate
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", rand_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = rand_search.best_params_
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
    'n_estimators': randint(50, 300),
    'learning_rate': loguniform(0.01, 0.2),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': loguniform(1e-2, 1.0),
    'reg_alpha': loguniform(1e-2, 1.0),
    'reg_lambda': loguniform(1.0, 10.0)
}


# Randomized search for XGBoost
rand_search = RandomizedSearchCV(
    estimator=XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    param_distributions={
        'n_estimators': randint(100, 300),
        'learning_rate': loguniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 6),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': loguniform(1e-2, 1.0),
        'reg_alpha': loguniform(1e-2, 1.0),
        'reg_lambda': loguniform(1.0, 10.0)
    },
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit model
rand_search.fit(X_train, y_train)

# Predict and evaluate
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", rand_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = rand_search.best_params_
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
    'n_estimators': randint(50, 300),
    'learning_rate': loguniform(0.01, 0.2),
    'estimator': [DecisionTreeRegressor(max_depth=d) for d in range(1, 4)]
}

# Randomized search for AdaBoost
rand_search = RandomizedSearchCV(
    estimator=AdaBoostRegressor(random_state=RANDOM_STATE),
    param_distributions={
        'n_estimators': randint(50, 300),
        'learning_rate': loguniform(0.01, 0.2),
        'estimator': [DecisionTreeRegressor(max_depth=d) for d in range(1, 4)]
    },
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit model
rand_search.fit(X_train, y_train)

# Predict and evaluate
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", rand_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = rand_search.best_params_
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
    'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64), (128, 128)],
    'alpha': loguniform(1e-5, 1e-2),
    'learning_rate_init': loguniform(1e-4, 1e-2),
    'learning_rate': ['constant', 'adaptive'],
    'activation': ['relu'],
    'solver': ['adam']
}


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Randomized search for MLP
rand_search = RandomizedSearchCV(
    estimator=MLPRegressor(random_state=RANDOM_STATE, max_iter=1000),
    param_distributions=param_grid,
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
rand_search.fit(X_train_scaled, y_train)

# Predict and evaluate
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", rand_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = rand_search.best_params_
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
    'num_leaves': randint(20, 100),
    'learning_rate': loguniform(0.01, 0.2),
    'n_estimators': randint(50, 300)
}

# Randomized search for LightGBM
rand_search = RandomizedSearchCV(
    estimator=LGBMRegressor(random_state=RANDOM_STATE),
    param_distributions={
        'num_leaves': randint(20, 100),
        'learning_rate': loguniform(0.01, 0.2),
        'n_estimators': randint(50, 300),
        'min_child_samples': randint(5, 30),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    },
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit model
rand_search.fit(X_train, y_train)

# Predict and evaluate
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", rand_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = rand_search.best_params_
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
    'depth': randint(4, 10),
    'learning_rate': loguniform(0.01, 0.2),
    'iterations': randint(100, 300),
    'l2_leaf_reg': loguniform(1, 10)
}


# Randomized search for CatBoost
rand_search = RandomizedSearchCV(
    estimator=CatBoostRegressor(
        verbose=0,
        random_state=RANDOM_STATE,
        loss_function='RMSE'
    ),
    param_distributions={
        'depth': randint(4, 10),
        'learning_rate': loguniform(0.01, 0.2),
        'iterations': randint(100, 300),
        'l2_leaf_reg': loguniform(1, 10)
    },
    n_iter=30,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1
)

# Fit model with categorical feature indices passed
rand_search.fit(X_train, y_train, cat_features=cat_features)

# Predict and evaluate
best_model = rand_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", rand_search.best_params_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = rand_search.best_params_
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

print(df_stats)

latex_ready = df_stats.copy()

for i, col in enumerate(df_stats.columns):
    if i < 3:
        latex_ready[col] = df_stats[col].apply(
            lambda x: f"{x:.1f}" if isinstance(x, (float, int)) else x
        )
    elif i == len(df_stats.columns) - 1:
        latex_ready[col] = df_stats[col].apply(
            lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x
        )

print(latex_ready.to_latex(index=True))


latex_ready.to_csv("Results/results_MLmodels2_pos.csv")



#%% Variable importance

# Get best model
best_mod_str = 'AdaBoost'  # override
best_model = best_models[best_mod_str]

# Get right X variable dataframes
X_train = X_train_ordinal.copy()
X_test = X_test_ordinal.copy()

# Save model
dump(best_model, "Models/best_mod_regression_pos.joblib")

# Feature importances
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    n_features_model = len(importances)
    n_features_data = X_test.shape[1]

    if n_features_model == n_features_data:
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(15)
    else:
        importance_df = pd.DataFrame({
            'Feature': [f"Feature_{i}" for i in range(n_features_model)],
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(15)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title(f"Top 15 Most Important Features ({best_mod_str})")
    plt.tight_layout()
    plt.savefig("Plots/feature_importance/feature_importance_drift_pos.png", dpi=500)
    plt.show()
else:
    print(f"No feature importances available for model: {best_mod_str}")

# SHAP with KernelExplainer
explainer = shap.KernelExplainer(best_model.predict, shap.kmeans(X_test, 10))
shap_values = explainer.shap_values(X_test, nsamples=1000)

# Global summary plot
shap.summary_plot(shap_values, X_test, show=True)

# Save plot
shap.summary_plot(shap_values, X_test, show=False)
fig = plt.gcf()
fig.tight_layout()
fig.savefig("Plots/feature_importance/feature_importance_shap_regression_pos.png", dpi=500, bbox_inches="tight")
fig.show()
plt.close(fig)



#%% OOS Analysis for AdaBoost

# Set y-variable
y_train_test = df_train_test["PostAbnormalReturn2Months"].copy()
y_simulation = df_simulation["PostAbnormalReturn2Months"].copy()

# Original feature sets
X_train_test_original = df_train_test.iloc[:, 2:-8].copy()
X_simulation_original = df_simulation.iloc[:, 2:-8].copy()

# Drop YearReport if present
for tt_set in [X_train_test_original, X_simulation_original]:
    if "YearReport" in tt_set.columns:
        tt_set.drop(columns="YearReport", inplace=True)

# Copy and encode using OrdinalEncoder (must match training)
X_train_test = X_train_test_original.copy()
X_simulation = X_simulation_original.copy()

string_cols = X_train_test.select_dtypes(include=["object", "string"]).columns

if len(string_cols) > 0:
    # Fit encoder on X_train_test
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(X_train_test[string_cols])
    
    # Transform both datasets
    X_train_test[string_cols] = encoder.transform(X_train_test[string_cols])
    X_simulation[string_cols] = encoder.transform(X_simulation[string_cols])

# Ensure same columns in same order
X_simulation = X_simulation[X_train_test.columns]

# Train AdaBoostRegressor with best hyperparameters
final_model = AdaBoostRegressor(
    **best_models_params["AdaBoost"],
    random_state=RANDOM_STATE
)

final_model.fit(X_train_test, y_train_test)

# Predict
y_sim_pred = final_model.predict(X_simulation)

# Compute metrics
mse = mean_squared_error(y_simulation, y_sim_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_simulation, y_sim_pred)
r2 = r2_score(y_simulation, y_sim_pred)

# Print results
print("Best Parameters:", best_models_params["AdaBoost"])
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R² Score:", r2)

# In-sample and OOS comparison table
stats_comparison = pd.DataFrame(best_models_stats["AdaBoost"], index=["In-Sample"])
stats_comparison.loc["Out-of-Sample"] = [mse, rmse, mae, r2]
stats_comparison.columns = ["MSE", "RMSE", "MAE", "R2"]
stats_comparison[["MSE", "RMSE", "MAE"]] = stats_comparison[["MSE", "RMSE", "MAE"]].round(1)
stats_comparison["R2"] = stats_comparison["R2"].round(2)
stats_comparison = stats_comparison.T

print(stats_comparison)
print(stats_comparison.to_latex())

# Prediction plot
plt.figure(figsize=(6,6))
plt.scatter(y_simulation, y_sim_pred, alpha=0.5)
plt.plot([y_simulation.min(), y_simulation.max()], [y_simulation.min(), y_simulation.max()], 'r--')
plt.xlabel("Actual 2-Month Post-Positive-Surprise Abnormal Return")
plt.ylabel("Predicted 2-Month Post-Positive-Surprise Abnormal Return")
plt.title("Out-of-Sample Prediction Accuracy (2024)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/error_analysis/prediction_plot_post.png")
plt.show()




#%% Simulation

sim_columns = [
    "QuarterReport",  "Industry", "SubIndustry",
    "PostAbnormalReturnSimulation", "PostRawReturnSimulation",
    "PostMarketReturnSimulation"
    ]

strategies = ["Drift", "Reversal", "Balanced"]


# Make returns df 
returns_sim = df_simulation[sim_columns].copy()

returns_sim["PostAbnormalReturnPredicted"] = y_sim_pred

returns_sim["TradeDrift"] = np.where(
    returns_sim["PostAbnormalReturnPredicted"] > 0, 1, 0
    )
returns_sim["TradeReversal"] = np.where(
    returns_sim["PostAbnormalReturnPredicted"] < 0, -1, 0
    )
returns_sim["TradeBalanced"] = returns_sim["TradeDrift"] + returns_sim["TradeReversal"]

# Add returns for each trading strategy
stats_summary = pd.DataFrame()
for strat in strategies:
    
    returns_sim[f"RealizedAbnormalReturn{strat}"] = returns_sim[f"Trade{strat}"] * returns_sim["PostAbnormalReturnSimulation"]
    returns_sim[f"RealizedRawReturn{strat}"] = returns_sim[f"Trade{strat}"] * returns_sim["PostRawReturnSimulation"]
    
    stats_summary[f"RealizedAbnormalReturn{strat}"] = returns_sim[returns_sim[f"Trade{strat}"] != 0][f"RealizedAbnormalReturn{strat}"].describe()
    stats_summary[f"RealizedRawReturn{strat}"] = returns_sim[returns_sim[f"Trade{strat}"] != 0][f"RealizedRawReturn{strat}"].describe()

# Active market return
returns_sim["MarketReturnActive"] = abs(returns_sim["TradeBalanced"]) * returns_sim["PostMarketReturnSimulation"]
stats_summary["MarketReturnActive"] = returns_sim[returns_sim["TradeBalanced"] != 0]["MarketReturnActive"].describe()

# Passive market return
mkt_return = pd.read_csv("Investing.com data/SP500 index.csv")
mkt_return["Year"] = pd.to_numeric(mkt_return["Date"].str[-4:])

mkt_return = mkt_return[mkt_return["Year"] == 2024]
mkt_return["Return"] = pd.to_numeric(mkt_return["Change %"].str.replace("%", ""))

stats_summary["MarketReturnPassive"] = mkt_return["Return"].describe()

# Finish summary stats
stats_summary = stats_summary.T
stats_summary["Sharpe Ratio"] = stats_summary["mean"] / stats_summary["std"]
stats_summary = stats_summary.round(2)
stats_summary["count"] = stats_summary["count"].astype(int)

print(stats_summary.to_latex())


# Density plot
surprise_returns = returns_sim[returns_sim["TradeDrift"] == 1][["RealizedAbnormalReturnDrift", "RealizedRawReturnDrift"]]

plt.figure(figsize=(8, 5))
sns.kdeplot(data=surprise_returns, x="RealizedAbnormalReturnDrift", label="Abnormal Return", fill=True)
sns.kdeplot(data=surprise_returns, x="RealizedRawReturnDrift", label="Raw Return", fill=True)
plt.title("Density Plot of Abnormal and Raw Returns")
plt.xlabel("Return (%)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/simulation/sim_returns_drift_pos.png", dpi=500)
plt.show()



