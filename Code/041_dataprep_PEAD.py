#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 16:06:38 2025

@author: juliusfrijns
"""

#%% Data prep for second analysis (PEAD)

# Imports
import numpy as np
import pandas as pd
import json
import statsmodels.api as sm

# Data import
with open("Panel data/sp500_panel_dtypes.json", "r") as f:
    dtypes = json.load(f)
df = pd.read_csv("Panel data/sp500_panel.csv", dtype=dtypes)
df = df.iloc[:, 1:]


# Data prep 

#%% Creation of earnings announcement dummies

# First remove all rows with NaN in price column
df = df[~df["PRICE"].isna()]

# Compute simple returns
df = df.sort_values(by=["Company", "Date"])
df["SimpleReturn"] = df.groupby("Company")["PRICE"].pct_change()


# Make column with quarter for later filtering and easyness
df["Quarter"] = np.nan
for q in ["Q1", "Q2", "Q3", "Q4"]:
    col = f"EARNINGS PER SHARE-REPRT DT-{q}"
    df["Quarter"] = np.where(df[col].notna(), q, df["Quarter"])

# Then make dummy vars for announcement days
df["Ann_str"] = df[[
    'EARNINGS PER SHARE-REPRT DT-Q1',
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
    ann_days = mask_df["Ann_str"].dropna().unique()

    # Add announcement dummies to mask df
    mask_df.loc[:, "ANNOUNCEMENT_DAY"] = np.where(mask_df["Date"].isin(ann_days), 1, 0)

    # Insert mask df into original df
    df.loc[df["Company"] == com, "ANNOUNCEMENT_DAY"] = mask_df["ANNOUNCEMENT_DAY"].values

    # Progress tracker
    print(com)


#%% Post-earnings period batch creation

# Now we create batches of each post-earnings period (quarter/2 quarters)

# Empty list for batch dfs to append toz
batches = []

# Loop through companies
for com in companies:
    
    # Create mask df and empty list for batch
    mask_df = df[df["Company"] == com].reset_index(drop=True)
    current_batch = []
    capture = False  # Flag to indicate whether we're currently capturing rows
    
    # Loop through all rows in the mask df
    for i in range(len(mask_df)):
        # If we hit an announcement day, start a new batch
        if mask_df.loc[i, "ANNOUNCEMENT_DAY"] == 1:
            # Save current batch if it's not empty and large enough
            if current_batch:
                batch_df = pd.DataFrame(current_batch)
                if len(batch_df) > 50:
                    batches.append(batch_df)
                current_batch = []  # Reset for the next batch
            capture = True  # Start capturing after this announcement
        
        # Capture all rows after an announcement day, including the announcement itself
        if capture:
            current_batch.append(mask_df.iloc[i])
    
    # After loop, add last batch if still valid
    if current_batch:
        batch_df = pd.DataFrame(current_batch)
        if len(batch_df) > 50:
            batches.append(batch_df)
    
    # Progress tracker
    print(com)

#%% Relevant variable creation

# Final storage
records = []

# Loop through all batches
for i, batch in enumerate(batches):
    
    # Cap the number of trading days a batch can have (to prevent way to long batches)
    if len(batch) > 67:
        batch = batch.iloc[:67]
        
    # Dependent variable: post-surprise abormal return
    # Abnormal returns
    y = batch["SimpleReturn"][1:] - batch["RF"][1:]
    X = sm.add_constant(batch[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]][1:])
    model = sm.OLS(y, X).fit()
    
    # Predict for this subset
    predicted = model.predict(X)
    residuals = y - predicted
    abnormal_return = residuals.values 
    
    post_abnormal_return_1week = np.nansum(abnormal_return[1:6])
    post_abnormal_return_1month = np.nansum(abnormal_return[1:22])
    post_abnormal_return_2months = np.nansum(abnormal_return[1:43])
    post_abnormal_return_3months = np.nansum(abnormal_return[1:64])
    
    # Date-based features
    dt = pd.to_datetime(batch["Date"].values[0])
    day_report = dt.day_name()
    year_report = dt.year
    quarter = batch["Quarter"].iloc[0] if "Quarter" in batch.columns else np.nan
    
    # Extract company name and ticker
    company = batch["Company"].iloc[0]
    ticker = batch["Ticker"].iloc[0]
    
    # Build record for final dataframe
    records.append({
        # Vars to merge on
        "Ticker": ticker,
        "QuarterReport": quarter,
        "YearReport": year_report,
        
        # Drift and excess drift
        "PostAbnormalReturn1Week": post_abnormal_return_1week,
        "PostAbnormalReturn1Month": post_abnormal_return_1month,
        "PostAbnormalReturn2Months": post_abnormal_return_2months,
        "PostAbnormalReturn3Months": post_abnormal_return_3months,
    })
    
    print(f"{i + 1}/{len(batches)}")
    
    
# Final DataFrame
df_batches = pd.DataFrame(records)

# Change dtype of YearReport as it needs to be categorical
df_batches["YearReport"] = df_batches["YearReport"].astype("object")

# Import other batch dataframe to be merged
df_pre_ann = pd.read_csv("Block data/batches_prepared.csv")

# Merge the datasets based on ticker and report date
df_final = pd.merge(
    df_pre_ann,
    df_batches,
    on=["Ticker", "YearReport", "QuarterReport"]
)
df_final = df_final.iloc[:, 1:]

# Export final df
df_final.to_csv("Block data/batches_postearn.csv")

# Save dtypes in json file as well
dtypes = df_final.dtypes.astype(str).replace("datetime64[ns]", "object")
dtypes.to_json("Block data/batches_postearn_dtypes.json")

# Final print
print("Block dataframe exported successfully")

