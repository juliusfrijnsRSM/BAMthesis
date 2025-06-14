#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:42:10 2025

@author: juliusfrijns
"""

# Part of thesis with time dependent train test split


#%% ML part of thesis

# Imports
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
#from sklearn.pipeline import Pipeline
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
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Define random state and keep constant
RANDOM_STATE = 678329 # <- my student number

# Set split year
split_year = 2022

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
df = df.dropna()

# Change reporting year to numeric
df["YearReport"] = pd.to_numeric(df["YearReport"])

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
df_train = df[df["YearReport"] < split_year]
df_test = df[df["YearReport"] >= split_year]

# Define y sets
y_train = df_train.iloc[:, -1]
y_test = df_test.iloc[:, -1]


# Original sets (no encoding or dimensionality reducing)
X_train_original = df_train.iloc[:, 2:-4]
X_test_original = df_test.iloc[:, 2:-4]


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
    X_test_ordinal[string_cols] = encoder.fit_transform(X_test_ordinal[string_cols])
    

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
for df in [
    X_train_original, X_test_original,
    X_train_reduced, X_test_reduced,
    X_train_ordinal, X_test_ordinal,
    X_train_label, X_test_label,
    X_train_onehot, X_test_onehot,
    X_train_onehot_reduced, X_test_onehot_reduced
]:
    if "YearReport" in df.columns:
        df.drop(columns="YearReport", inplace=True)


# Remapping for multiclass modelling
y_train = y_train + 1
y_test = y_test + 1
    
# Create empty dict to append best model to
best_models = dict()
best_models_params = dict()
best_models_stats = dict()


#%% Logistic model

model_name = "Logistic"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot_reduced.copy()
X_test = X_test_onehot_reduced.copy()

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

# Randomized search parameters (flattened structure)
param_distributions = [
    {'clf__penalty': ['l1'], 'clf__C': [0.01, 0.1, 1.0]},
    {'clf__penalty': ['l2'], 'clf__C': [0.01, 0.1, 1.0]},
    {'clf__penalty': ['elasticnet'], 'clf__C': [0.01, 0.1, 1.0], 'clf__l1_ratio': [0.0, 0.5, 1.0]}
]


# Randomized search
random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    scoring=custom_scorer,
    cv=5,
    n_iter=15,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)

# Fit model
random_search.fit(X_train, y_train)

# Predict and evaluate
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", random_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = report


#%% KNN model

model_name = "KNN"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot_reduced.copy()
X_test = X_test_onehot_reduced.copy()

# Tuning grid (used as param_distributions)
param_distributions = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['minkowski'],
    'knn__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

# Pipeline
pipe = Pipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Randomized search
random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=150,
    scoring=custom_scorer,
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

# Print results
print("Best Parameters:", random_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = report


#%% Random forest model

model_name = "Random Forest"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_ordinal.copy()
X_test = X_test_ordinal.copy()

# Parameter distributions for randomized search
param_distributions = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Define model
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE
)

# Randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=150,
    scoring=custom_scorer,
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

# Print results
print("Best Parameters:", random_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = report


#%% XGBoosting model

model_name = "XGBoost"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_ordinal.copy()
X_test = X_test_ordinal.copy()

# Parameter distributions for randomized search
param_distributions = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 8],
    'min_child_weight': [1, 2, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.5, 1],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0.1, 1, 5]
}

# Define model
model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=150,
    scoring=custom_scorer,
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

# Print results
print("Best Parameters:", random_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = report


#%% AdaBoost model

model_name = "AdaBoost"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_ordinal.copy()
X_test = X_test_ordinal.copy()

# Define base estimators
base_estimators = [
    DecisionTreeClassifier(max_depth=i, class_weight='balanced') for i in range(1, 4)
]

# Define tuning distributions (discrete values)
param_distributions = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'learning_rate': [0.005, 0.01, 0.025, 0.05, 0.1],
    'estimator': base_estimators
}

# Define model
model = AdaBoostClassifier(random_state=RANDOM_STATE)

# Define randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=150,
    scoring=custom_scorer,
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

# Print results
print("Best Parameters:", random_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = report


#%% MLP model

model_name = "MLP"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot_reduced.copy()
X_test = X_test_onehot_reduced.copy()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute sample weights manually
sample_weights = compute_sample_weight(class_weight={0: 3, 1: 1, 2: 3}, y=y_train)

# Define parameter distributions (discrete)
param_distributions = {
    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 128)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [1000],
}

# Grid search without sample_weight
mlp = MLPClassifier(early_stopping=True, random_state=RANDOM_STATE)
random_search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_distributions,
    n_iter=150,
    scoring=custom_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Fit model (no class weights yet)
random_search.fit(X_train_scaled, y_train)

# Refit best model with sample weights
best_model = MLPClassifier(
    **random_search.best_params_,
    early_stopping=True,
    random_state=RANDOM_STATE
)
best_model.fit(X_train_scaled, y_train)  # , sample_weight=sample_weights

# Predict and evaluate
y_pred = best_model.predict(X_test_scaled)

# Print results
print("Best Parameters:", random_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = report


#%% LightGBM model

model_name = "LightGBM"
print(f"\n{model_name}")

# Use the same features
X_train = X_train_label.copy()
X_test = X_test_label.copy()

# Sample weights
sample_weights = compute_sample_weight(class_weight={0: 3, 1: 1, 2: 3}, y=y_train)

# Define LightGBM parameter distributions (discrete)
param_distributions = {
    'num_leaves': [15, 31, 63, 127],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'n_estimators': [100, 150, 200, 250],
    'class_weight': [None]  # Using manual sample_weight
}

# Initialize base model
lgbm = LGBMClassifier(random_state=RANDOM_STATE)

# Randomized search
random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_distributions,
    n_iter=150,
    scoring=custom_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Fit with sample weights
random_search.fit(X_train, y_train, sample_weight=sample_weights)

# Refit best model
best_model = LGBMClassifier(
    **random_search.best_params_,
    random_state=RANDOM_STATE
)
best_model.fit(X_train, y_train, sample_weight=sample_weights)

# Predict and evaluate
y_pred = best_model.predict(X_test)
print("Best Parameters:", random_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = report


#%% CatBoost model

model_name = "CatBoost"
print(f"\n{model_name}")

# Use correct train/test set
X_train = X_train_original.copy()
X_test = X_test_original.copy()

# Get categorical column indices from names (assumes string_cols = list of column names)
cat_features = [X_train.columns.get_loc(col) for col in list(string_cols)]

# Convert those columns to string to satisfy CatBoost
for col in string_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# Sample weights
sample_weights = compute_sample_weight(class_weight={0: 3, 1: 1, 2: 3}, y=y_train)

# Define hyperparameter distributions (discrete)
param_distributions = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [200, 300, 400],
    'l2_leaf_reg': [1, 3, 5, 7]
}

# Define the base model
catboost = CatBoostClassifier(
    verbose=0,
    random_state=RANDOM_STATE,
    loss_function='MultiClass'
)

# Prepare fit_params to pass cat_features and sample_weight
fit_params = {
    'cat_features': cat_features,
    'sample_weight': sample_weights
}

# Run randomized search
random_search = RandomizedSearchCV(
    estimator=catboost,
    param_distributions=param_distributions,
    n_iter=150,
    scoring=custom_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

random_search.fit(X_train, y_train, **fit_params)

# Refit best model
best_model = CatBoostClassifier(
    **random_search.best_params_,
    verbose=0,
    random_state=RANDOM_STATE,
    loss_function='MultiClass'
)

best_model.fit(X_train, y_train, sample_weight=sample_weights, cat_features=cat_features)

# Predict and evaluate
y_pred = best_model.predict(X_test)
print("Best Parameters:", random_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Store results
best_models[model_name] = best_model
best_models_params[model_name] = random_search.best_params_
best_models_stats[model_name] = report


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
    
    # Custom score (weighed by class 0 and 2 only)
    f1_0 = stats['0']['f1-score']
    f1_2 = stats['2']['f1-score']
    s_0 = stats['0']['support']
    s_2 = stats['2']['support']
    
    # Custom weighted F1 for classes 0 and 2 only
    f1_custom = (f1_0 * s_0 + f1_2 * s_2) / (s_0 + s_2)
    
    row = {
        'Model': model_name,
        'Custom F1': f1_custom,
        'F1 (0)': stats['0']['f1-score'],
        'Precision (0)': stats['0']['precision'],
        'Recall (0)': stats['0']['recall'],
        'F1 (2)': stats['2']['f1-score'],
        'Precision (2)': stats['2']['precision'],
        'Recall (2)': stats['2']['recall'],
        'F1 (Class 1)': stats['1']['f1-score'],
        'Precision (Class 1)': stats['1']['precision'],
        'Recall (Class 1)': stats['1']['recall'],
        'Accuracy': accuracy,
        'Macro F1': macro['f1-score'],
        'Macro Precision': macro['precision'],
        'Macro Recall': macro['recall'],
        'Weighted F1': weighted['f1-score'],
        'Weighted Precision': weighted['precision'],
        'Weighted Recall': weighted['recall'],
    }
    rows.append(row)

df_stats = pd.DataFrame(rows)

# Sort based on weighted F1
df_stats = df_stats.sort_values(by='Custom F1', ascending=False)
df_stats.index = df_stats["Model"]
df_stats = df_stats.drop(columns=["Model"])

print(df_stats.iloc[:, :7])
latex_ready = df_stats.iloc[:, :7].map(lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, float) else x)
print(latex_ready.to_latex())


#%% Variable importance

# Store best model
best_mod_str = df_stats.index[0]
best_model = best_models[best_mod_str]

# Save best model externally to prevent having to tune and train all the time
dump(best_model, "Models/best_mod_classification.joblib")

#### Next part could be not future proof as it might be for CatBoost specifically


# Get feature importance values and feature names
importances = best_model.get_feature_importance()
features = best_model.feature_names_

# Create a dataframe and sort
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)

# Plot
plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Top 15 Most Important Features (CatBoost)")
plt.tight_layout()
plt.savefig("Plots/feature_importance/feature_importance_surprise.png")
plt.show()


# Create explainer
explainer = shap.Explainer(best_model)

# Compute SHAP values
shap_values = explainer(X_test)

# Global summary plot
shap.summary_plot(shap_values, X_test)

# Loop and show
for i in range(3):
    
    # To show inline
    shap.summary_plot(
        shap_values[..., i], 
        X_test, 
        show=True, 
        plot_type="dot", 
        max_display=20
    )
    
    # To save in folder
    shap.summary_plot(
        shap_values[..., i], 
        X_test, 
        show=False, 
        plot_type="dot", 
        max_display=20
    )
    
    fig = plt.gcf() 
    fig.tight_layout()
    fig.savefig(f"Plots/feature_importance/feature_importance_shap_class{i}.png")
    fig.show()
    plt.close(fig)  # Close it to suppress display in loop

