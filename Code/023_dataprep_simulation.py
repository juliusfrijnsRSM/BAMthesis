#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 21:02:33 2025

@author: juliusfrijns
"""

#%% Data prep for simulation

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

# Empty list for batch dfs to append to
batches = []

for com in companies:
    mask_df = df[df["Company"] == com].reset_index(drop=True)
    announcement_indices = mask_df.index[mask_df["ANNOUNCEMENT_DAY"] == 1].tolist()

    # Skip if less than 2 announcements – not enough for a full batch
    if len(announcement_indices) < 2:
        continue

    # Iterate over triples of announcements: t-1, t, t+1
    for i in range(1, len(announcement_indices) - 1):
        t_minus_1 = announcement_indices[i - 1]
        t = announcement_indices[i]
        t_plus_1 = announcement_indices[i + 1]

        # Start the batch the day after t-1 and end the day before t+1
        batch_df = mask_df.iloc[t_minus_1 + 1 : t_plus_1].copy()
        if len(batch_df) > 50:
            batches.append(batch_df)

    print(com)

#%% Relevant variable creation

# Final storage
records = []

# Loop through all batches
for i, batch in enumerate(batches):
    
    # Define pre- and post-announcement batches
    ann_idx = batch[batch["ANNOUNCEMENT_DAY"] == 1].index[0]
    batch_pre = batch.loc[:ann_idx, :]
    batch_post = batch.loc[ann_idx:, :]
    
    # Cap the number of trading days a batch can have (to prevent way to long batches)
    if len(batch_pre) > 66:
        batch_pre = batch_pre.iloc[-66:]
        
    if len(batch_post) > 67:
        batch_post = batch_post.iloc[:67]
        
    # Check if batches are actually large enough
    if len(batch_pre) < 20 or len(batch_post) < 20:
        continue

    # Fit model on pre-announcement returns
    y_pre = batch_pre["SimpleReturn"] - batch_pre["RF"]
    X_pre = sm.add_constant(batch_pre[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    model = sm.OLS(y_pre, X_pre).fit()
    
    # Predict for this subset
    y_post = batch_post["SimpleReturn"][1:] - batch_post["RF"][1:]
    X_post = sm.add_constant(batch_post[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]][1:])
    
    predicted = model.predict(X_post)
    residuals = y_post - predicted
    abnormal_return = residuals.values
    
    post_abnormal_return_1week = np.nansum(abnormal_return[:5]) * 100
    post_abnormal_return_1month = np.nansum(abnormal_return[:21]) * 100
    post_abnormal_return_2months = np.nansum(abnormal_return[:42]) * 100
    post_abnormal_return_3months = np.nansum(abnormal_return[:63]) * 100
    
    # Simulation var
    post_abnormal_return_1day = abnormal_return[0] * 100
    
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
        
        "PostAbnormalReturn1Day": post_abnormal_return_1day,
    })
    
    print(f"{i + 1}/{len(batches)}")
    
    
# Final DataFrame
df_batches = pd.DataFrame(records)

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
df_final.to_csv("Block data/batches_simulation.csv")

# Save dtypes in json file as well
dtypes = df_final.dtypes.astype(str).replace("datetime64[ns]", "object")
dtypes.to_json("Block data/batches_simulation.json")

# Final print
print("Block dataframe exported successfully")

