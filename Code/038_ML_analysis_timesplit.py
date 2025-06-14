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
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
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
split_year1 = 2022
split_year2 = 2024

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

# Delete infinite rows
numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df[~np.isinf(df[numeric_cols]).any(axis=1)]

# Columns to drop for reduced dimensionality X sets
cols_to_drop = [
    #"YearReport",
    "Country",
    "Industry",
    "SubIndustry",
]

# Define train and test dataframes based on year to split on
df_train = df[df["YearReport"] < split_year1]
df_test = df[(df["YearReport"] >= split_year1) & (df["YearReport"] < split_year2)]

df_train_test = df[df["YearReport"] < split_year2]
df_simulation = df[df["YearReport"] >= split_year2]

# Drop simulation column as it should not be used in model training
sim_cols_to_drop = [
    "AbnormalReturnSimulation", "RawReturnSimulation",
    "MarketReturnSimulation"
    ]

df_train = df_train.drop(columns = sim_cols_to_drop)
df_test = df_test.drop(columns = sim_cols_to_drop)
df_train_test = df_train_test.drop(columns = sim_cols_to_drop)

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
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = report


#%% KNN model

model_name = "KNN"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_onehot_reduced.copy()
X_test = X_test_onehot_reduced.copy()

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
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = report


#%% Random forest model

model_name = "Random Forest"
print(f"\n{model_name}")

# Get right X variable dataframes
X_train = X_train_ordinal.copy()
X_test = X_test_ordinal.copy()

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
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = report


#%% XGBoosting model

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
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
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
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)


# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = report


#%% MLP model

# Model name
model_name = "MLP"
print(f"\n{model_name}")

# Use appropriate X and y
X_train = X_train_onehot_reduced.copy()
X_test = X_test_onehot_reduced.copy()

# Define param grid
param_grid = {
    'mlp__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 128)],
    'mlp__activation': ['relu'],
    'mlp__solver': ['adam'],
    'mlp__alpha': [1e-5, 1e-4, 1e-3],
    'mlp__learning_rate': ['constant', 'adaptive'],
    'mlp__learning_rate_init': [0.001, 0.01],
    'mlp__max_iter': [1000],
}

# Pipeline with SMOTE and scaling
pipe = Pipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(early_stopping=True, random_state=RANDOM_STATE))
])

# Grid search with custom scorer
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=custom_scorer,
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
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Store results
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
best_models_stats[model_name] = report


#%% LightGBM model

model_name = "LightGBM"
print(f"\n{model_name}")

# Use the same features
X_train = X_train_label.copy()
X_test = X_test_label.copy()

# Sample weights
sample_weights = compute_sample_weight(class_weight={0: 3, 1: 1, 2: 3}, y=y_train)

# Define LightGBM parameter grid
param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'class_weight': [None]  # We use manual sample_weight instead
}

# Initialize model
lgbm = LGBMClassifier(random_state=RANDOM_STATE)

# Grid search
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Fit best model with sample weights
best_model = LGBMClassifier(
    **grid_search.best_params_,
    random_state=RANDOM_STATE
)
best_model.fit(X_train, y_train, sample_weight=sample_weights)

# Predict and evaluate
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Append
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
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

# Define hyperparameter grid
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1],
    'iterations': [300],
    'l2_leaf_reg': [1, 3, 5]
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

# Run grid search
grid_search = GridSearchCV(
    estimator=catboost,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train, **fit_params)

# Refit best model
best_model = CatBoostClassifier(
    **grid_search.best_params_,
    verbose=0,
    random_state=RANDOM_STATE,
    loss_function='MultiClass'
)

best_model.fit(X_train, y_train, sample_weight=sample_weights, cat_features=cat_features)

# Predict and evaluate
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0))
report = classification_report(y_test, y_pred, labels=[0, 1, 2], zero_division=0, output_dict=True)

# Store results
best_models[model_name] = best_model
best_models_params[model_name] = grid_search.best_params_
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

latex_ready.to_csv("Results/results_MLmodels1.csv")


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
plt.savefig("Plots/feature_importance/feature_importance_surprise.png", dpi=500)
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
    fig.savefig(f"Plots/feature_importance/feature_importance_shap_class{i}.png", dpi=500)
    fig.show()
    plt.close(fig)  # Close it to suppress display in loop


#%% Final model training and simulation evaluation (CatBoost + SMOTE)


# Prepare train/simulation sets
X_train_test = df_train_test.iloc[:, 2:-4].copy()
X_simulation = df_simulation.iloc[:, 2:-4].copy()
y_train_test = df_train_test.iloc[:, -1].copy()
y_simulation = df_simulation.iloc[:, -1].copy()

# Re-do the class labelling
y_train_test = y_train_test + 1
y_simulation = y_simulation + 1

# Drop YearReport if present
for df_ in [X_train_test, X_simulation]:
    if "YearReport" in df_.columns:
        df_.drop(columns="YearReport", inplace=True)

# Identify and convert string columns
string_cols = X_train_test.select_dtypes(include=["object", "string"]).columns
for col in string_cols:
    X_train_test[col] = X_train_test[col].astype(str)
    X_simulation[col] = X_simulation[col].astype(str)

cat_features = [X_train_test.columns.get_loc(col) for col in string_cols]

# Create encoded version for SMOTE
X_encoded = X_train_test.copy()
for col in string_cols:
    X_encoded[col] = X_encoded[col].astype("category").cat.codes

# Apply SMOTE
X_balanced, y_balanced = SMOTE(random_state=678329).fit_resample(X_encoded, y_train_test)

# Sample weights on SMOTE-augmented labels
sample_weights = compute_sample_weight(class_weight={0: 3, 1: 1, 2: 3}, y=y_balanced)

# Create aligned string-based feature DataFrame for CatBoost
X_catboost = pd.DataFrame(
    X_balanced, columns=X_encoded.columns
).copy()
for col in string_cols:
    X_catboost[col] = X_catboost[col].astype("int").map(
        dict(enumerate(X_train_test[col].astype("category").cat.categories))
    ).astype(str)

# Train final CatBoost model
final_model = CatBoostClassifier(
    **best_models_params[best_mod_str],
    verbose=0,
    random_state=678329,
    loss_function='MultiClass'
)

final_model.fit(X_catboost, y_balanced, cat_features=cat_features, sample_weight=sample_weights)

# Predict on 2024 simulation set
y_sim_pred = final_model.predict(X_simulation)

# Evaluate performance
print("Classification Report on Simulation Set (2024):")
print(classification_report(y_simulation, y_sim_pred, labels=[0, 1, 2], zero_division=0))

# Create df of confusion matrix
sim_report_df = pd.DataFrame(
    classification_report(y_simulation, y_sim_pred, labels=[0, 1, 2], output_dict=True, zero_division=0)
    ).T.round(2) # round to 2 decimals

# Make support column int type
sim_report_df["support"] = sim_report_df["support"].round(0).astype(int) 

# Print to latex
print(sim_report_df)
print(sim_report_df.to_latex())

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_simulation, y_sim_pred, display_labels=["Pos", "Neutral", "Neg"])
plt.title("Confusion Matrix – Simulation Set (2024)")
plt.savefig("Plots/error_analysis/conf_matrix.png", dpi=500)


#%% Simulation

sim_columns = [
    "QuarterReport",  "Industry", "SubIndustry",
    "AbnormalReturnSimulation", "RawReturnSimulation",
    "MarketReturnSimulation",
    "SurpriseDir"
    ]

# Make returns df 
returns_sim = df_simulation[sim_columns].copy()
returns_sim["SurprisePred"] = y_sim_pred.flatten() - 1

returns_sim["RealizedAbnormalReturn"] = returns_sim["SurprisePred"] * returns_sim["AbnormalReturnSimulation"]
returns_sim["RealizedRawReturn"] = returns_sim["SurprisePred"] * returns_sim["RawReturnSimulation"]
returns_sim["MarketReturnActive"] = abs(returns_sim["SurprisePred"]) * returns_sim["MarketReturnSimulation"]


# Summary table of simulation statistics
stats_cols = ["RealizedAbnormalReturn", "RealizedRawReturn", "MarketReturnActive"]
stats_summary = returns_sim[returns_sim["SurprisePred"] != 0][stats_cols].describe()

# Import market returns
mkt_return = pd.read_csv("Investing.com data/SP500 index.csv")
mkt_return["Year"] = pd.to_numeric(mkt_return["Date"].str[-4:])

mkt_return = mkt_return[mkt_return["Year"] == 2024]
mkt_return["Return"] = pd.to_numeric(mkt_return["Change %"].str.replace("%", ""))

stats_summary["MarketReturnPassive"] = mkt_return["Return"].describe()

stats_summary = stats_summary.T

stats_summary["Sharpe Ratio"] = stats_summary["mean"] / stats_summary["std"]
stats_summary = stats_summary.round(2)
stats_summary["count"] = stats_summary["count"].astype(int)
print(stats_summary.to_latex())

# Density plot
surprise_returns = returns_sim[returns_sim["SurprisePred"] != 0][["RealizedAbnormalReturn", "RealizedRawReturn"]]

plt.figure(figsize=(8, 5))
sns.kdeplot(data=surprise_returns, x="RealizedAbnormalReturn", label="Abnormal Return", fill=True)
sns.kdeplot(data=surprise_returns, x="RealizedRawReturn", label="Raw Return", fill=True)
plt.title("Density Plot of Abnormal and Raw Returns")
plt.xlabel("Return (%)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/simulation/sim_returns_density.png", dpi=500)
plt.show()

