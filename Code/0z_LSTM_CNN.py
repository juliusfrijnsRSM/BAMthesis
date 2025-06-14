#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:33:46 2025

@author: juliusfrijns
"""

#%% Data prep for ML analysis with time series neural nets

# Imports
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

# Data import
with open("Panel data/sp500_panel_dtypes.json", "r") as f:
    dtypes = json.load(f)
df = pd.read_csv("Panel data/sp500_panel.csv", dtype=dtypes)
df = df.iloc[:, 1:]

# Safe log function
def safe_log(series):
    # Only take log of positive and non-null values
    return np.log(series.where(series > 0))


#%% Creation of earnings announcement dummies

# First remove all rows with NaN in price column
df = df[~df["PRICE"].isna()]

# Make column with quarter for later filtering and easyness
df["Quarter"] = np.nan

for q in ["Q1", "Q2", "Q3", "Q4"]:
    col = f"EARNINGS PER SHARE-REPRT DT-{q}"
    df["Quarter"] = np.where(df[col].notna(), q, df["Quarter"])

# Then make dummy vars for announcement days
df["Ann_str"] = df[['EARNINGS PER SHARE-REPRT DT-Q1',
                    'EARNINGS PER SHARE-REPRT DT-Q2', 
                    'EARNINGS PER SHARE-REPRT DT-Q3',
                    'EARNINGS PER SHARE-REPRT DT-Q4']].bfill(axis=1).iloc[:, 0]

# List of companies in dataset
companies = list(df["Company"].unique())

df["ANNOUNCEMENT_DAY"] = np.nan
for com in companies:
    
    # Create mask df based on company
    mask_df = df[df["Company"] == com]
    
    # Create array of unique announcement days
    ann_days = mask_df["Ann_str"].unique()
    
    # Add announcement dummies to mask df
    mask_df.loc[:, "ANNOUNCEMENT_DAY"] = np.where(mask_df["Date"].isin(ann_days), 1, 0)
    
    # Insert mask df into original df
    df.loc[df["Company"] == com, "ANNOUNCEMENT_DAY"] = mask_df["ANNOUNCEMENT_DAY"].values
    
    # Progress tracker
    print(com)


#%% Earnings period batch creation


# Empty list for batch dfs to append to
batches = []

# Loop through each company group
for com, group in df.groupby("Company"):
    group = group.reset_index(drop=True)
    
    # Find all announcement days
    ann_idx = group.index[group["ANNOUNCEMENT_DAY"] == 1].tolist()

    for idx in ann_idx:
        # Batch includes the current row and the next one
        batch = group.iloc[max(0, idx - 60): idx + 2]  # ensures at most 62 rows
        
        if len(batch) >= 62:
            batch = batch.iloc[-62:]  # ensure it's exactly 62 rows

            # Vectorized feature engineering
            batch = batch.copy()
            batch["LogReturn"] = safe_log(batch["PRICE"] / batch["PRICE"].shift())
            batch["LogReturnIndex"] = safe_log(batch["CloseSP500"] / batch["CloseSP500"].shift())
            batch["ExcessReturn"] = batch["LogReturn"] - batch["LogReturnIndex"]
            
            batch["HiLowGap"] = (batch["PRICE HIGH"] - batch["PRICE LOW"]) / batch["PRICE"]
            batch["LogVolume"] = safe_log(batch["TURNOVER BY VOLUME"])
            
            batch["LogVolumeIndex"] = safe_log(batch["VolumeSPY"])
            batch["HiLowGapIndex"] = (batch["HighSP500"] - batch["LowSP500"]) / batch["CloseSP500"]
            
            batch["LogBrent"] = safe_log(batch["DCOILBRENTEU"])
            batch["LogWTI"] = safe_log(batch["DCOILWTICO"])
            batch["LogVIX"] = safe_log(batch["VIXCLS"])

            # Compute surprise
            hist_vol = np.nanstd(batch["ExcessReturn"].iloc[:-1])
            final_return = batch["ExcessReturn"].iloc[-1]
            
            if final_return > 1.96 * hist_vol:
                surprise = 2
            elif final_return < -1.96 * hist_vol:
                surprise = 0
            else:
                surprise = 1

            batch["Surprise"] = surprise

            # Select relevant columns
            relevant_cols = [
                "ExcessReturn",
                "HiLowGap",
                "LogVolume",
                "LogReturnIndex",
                "HiLowGapIndex",
                "LogVolumeIndex",
                "LogBrent",
                "LogWTI",
                "LogVIX",
                "Inflation rate",
                "FEDFUNDS",
                "REAINTRATREARAT1MO",
                "REAINTRATREARAT1YE",
                "REAINTRATREARAT10Y",
                "UNRATE",
                "Surprise"
            ]

            batches.append(batch[relevant_cols].iloc[-61:-1])  # last 60 observations only and prevent leakage

    print(com)  # progress tracker


#%% Data prep for tensorflow

# Set variables
X = np.array([batch.iloc[:, :-1].to_numpy() for batch in batches])
y = np.array([batch.iloc[-1, -1] for batch in batches])

print(X.shape)  # Expect (samples, timesteps, features)
print(y.shape)  # Expect (samples,)
print(np.unique(y))

y_cat = to_categorical(y, num_classes=3)  # shape: (samples, 3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

#%% Model (tuning)

# 🔹 STEP 1: Compute class weights (use original class labels)
y_train_int = y_train.argmax(axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)
class_weight_dict = dict(enumerate(class_weights))


# 🔹 STEP 2: Define the model for KerasTuner
def build_model(hp):
    model = Sequential()
    
    model.add(LSTM(
        units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32),
        return_sequences=True,
        input_shape=(X.shape[1], X.shape[2])
    ))
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))
    
    model.add(LSTM(
        units=hp.Int('lstm_units_2', min_value=16, max_value=64, step=16)
    ))
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))

    model.add(Dense(hp.Int('dense_units', 16, 64, step=16), activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# 🔹 STEP 3: Setup the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='kt_logs',
    project_name='lstm_tuning'
)

# 🔹 STEP 4: Run the search with class_weight
tuner.search(
    X_train, y_train,
    epochs=20,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
    class_weight=class_weight_dict
)

# 🔹 STEP 5: Retrieve the best model and hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(1)[0]

print("Best hyperparameters:")
for key in best_hps.values:
    print(f"{key}: {best_hps.get(key)}")





#%% Model (singular)

model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)


y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, digits=3))
