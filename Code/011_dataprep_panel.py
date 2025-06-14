#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:04:09 2025

@author: juliusfrijns

Data prep script for BAM Thesis
"""

#%% Data prep script for BAM Thesis

# Imports
import numpy as np
import pandas as pd
import json
import warnings

#%% Start-End sheets

# Constant variables
FIRST_SHEET = 1
LAST_SHEET = 16

num_of_cols = { # Keeps track of thenumber of excess columns each sheet should have in the end
    1: 3, 
    2: 3, 
    3: 3, 
    4: 3,
    5: 2,
    6: 3,
    7: 2,
    8: 3,
    9: 3,
    10: 3,
    11: 3,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 3
    }

mapping_sp500 = pd.read_csv("Mappings/SP500_mapping.csv")

# Make into function to save memory

def transform_panel(df, company_mapping):
    
    # Transform date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    
    # Drop the error column(s)
    number_error_cols = len(df.loc[:, df.columns.str.startswith('#ERROR')].columns)
    print(f"Number of #ERROR columns: {number_error_cols}")
    df = df.loc[:, ~df.columns.str.startswith('#ERROR')].copy()
    
    
    # Convert any columns in object form to datetime to prevent agg error later
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Melt the dataframe to get better format for panel data (tbc)
    df_long = df.melt(id_vars="Date", var_name="Raw_Column", value_name="Value")
    
    # Change column names to fix edge case CURRENT ASSETS - TOTAL in next step
    df_long["Raw_Column"] = df_long["Raw_Column"].str.replace("CURRENT ASSETS - TOTAL", "CURRENT ASSETS", regex=False)

    # Split the raw column names into company and variable
    df_long[["Company", "Variable"]] = df_long["Raw_Column"].str.rsplit(" - ", n=1, expand=True)

    # Create new panel df with only relevant columns (for now)
    df_long = df_long[['Date', 'Company', 'Variable', 'Value']]
    
    panel_df = df_long.drop_duplicates(subset=["Date", "Company"], keep="first")
    panel_df = panel_df[["Date", "Company"]]
    
    for var in df_long["Variable"].unique():
        df_var = df_long.loc[df_long["Variable"] == var, ["Date", "Company", "Value"]].copy()
        df_var = df_var.rename(columns={"Value": var})
        
        # Create sample for checking numeric and datetime
        sample = df_var[var].dropna().astype(str).head(10)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            
            # Try parsing to datetime
            dt_parsed = pd.to_datetime(sample, errors='coerce')
        
        # Check if parsed to datetime correctly
        if dt_parsed.notna().mean() > 0.8:
            df_var[var] = pd.to_datetime(df_var[var], errors='coerce')
            print(f"{var} → converted to datetime")

        else:
            # Try parsing to numeric
            num_parsed = pd.to_numeric(sample, errors='coerce')
            
            # Checkif parsed to numeric correctly
            if num_parsed.notna().mean() > 0.8:
                df_var[var] = pd.to_numeric(df_var[var], errors='coerce')
                print(f"{var} → converted to numeric")
                
            # If not datetime or numeric leave as object
            else:
                print(f"{var} → left as object")
        
        panel_df = pd.merge(
            panel_df, 
            df_var, 
            on=["Date", "Company"], 
            how="left"
        )
        
    
    return panel_df




# Execute function

allsheets = []
for i in range(FIRST_SHEET, LAST_SHEET + 1):
    
    print(f"Transforming sheet {i}...")
    
    # Load dtypes from JSON
    with open(f"Datastream CSVs/SP500 - {i}_dtypes.json", "r") as f:
        dtypes = json.load(f)
    
    # Import CSV file
    df = pd.read_csv(f"Datastream CSVs/SP500 - {i}.csv", dtype=dtypes)
    
    # Execute function
    temp_df = transform_panel(df, mapping_sp500)
    
    # Keep only relevant columns
    temp_df = temp_df.iloc[:, :2+num_of_cols[i]]
    
    # Append to list
    allsheets.append(temp_df)
    
    # Merge the sheets
    if i == 1:
        panel_df = temp_df.copy()
        
    else:
        panel_df = pd.merge(
            panel_df,
            temp_df,
            on=["Date", "Company"],
            how="outer"
        )
        
    
    print(f"Sheet {i} finished.\n")

panel_df = panel_df.sort_values(by=["Company", "Date"])



# Apply company mapping to the df
# This was not done inside the function because of memory issues
# Later the df will be collapsed
mapping_dict = dict(zip(mapping_sp500["Original"], mapping_sp500["Mapped To"]))

companies = panel_df["Company"].tolist()
mapped_companies = [
    mapping_dict.get(company, company)
    for company in companies
]

# Change companies to mapped counterparts to prevent duplicates
panel_df["Company"] = mapped_companies

panel_df = panel_df.groupby(["Date", "Company"], as_index=False).first().sort_values(by=["Company", "Date"]).reset_index(drop=True)


#%% As of (Static) sheets

# New column names (not Datastream format)
STATIC_COLUMNS = [
    "Company",
    "Ticker",
    "ISIN",
    "GeoCode",
    "GeoLocation",
    "Industry",
    "SubIndustry",
    "Type",
    "StockType",
    "CURRENCY"
]

# Import excel file
sp500_static = pd.read_excel(
    "Datastream/Datastream static HC.xlsx", 
    sheet_name="SP500 HC"
    ).T.reset_index(drop=True)

# Clean up top of dataframe
sp500_static.columns = STATIC_COLUMNS
sp500_static = sp500_static.iloc[1:]

# Clean ticker string
sp500_static["Ticker"] = sp500_static["Ticker"].str.replace("@:", "")
sp500_static["Ticker"] = sp500_static["Ticker"].str.replace("1", "")

# Drop duplicate Google/Alphabet ticker
sp500_static = sp500_static[sp500_static["Ticker"] != "GOOGM"]

# Import static mapping
mapping_static = pd.read_csv("Mappings/SP500_mapping_static.csv")
mapping_static = mapping_static.iloc[:, 1:]
mapping_dict_static = dict(zip(mapping_static["Panel"], mapping_static["Static"]))

companies = panel_df["Company"].tolist()
mapped_companies = [
    mapping_dict_static.get(company, company)
    for company in companies
]

panel_df["Company"] = mapped_companies
panel_df = panel_df.groupby(["Date", "Company"], as_index=False).first().sort_values(by=["Company", "Date"]).reset_index(drop=True)

# Merge with static variables
panel_df = pd.merge(sp500_static, panel_df, how="right", on="Company")
panel_df = panel_df[~panel_df["Ticker"].isna()]



#%% FRED data (static)


# Daily: DCOILBRENTEU, DCOILWTICO, VIXCLS
# Monthly: CPIAUCSL, FEDFUNDS, REAINTRATREARAT1MO, REAINTRATREARAT1YE, REAINTRATREARAT10Y, UNRATE
# Quarterly: GDP

daily_datasets = [
    "DCOILBRENTEU", 
    "DCOILWTICO", 
    "VIXCLS"
]

monthly_datasets = [
    "CPIAUCSL",
    "FEDFUNDS",
    "REAINTRATREARAT1MO",
    "REAINTRATREARAT1YE",
    "REAINTRATREARAT10Y",
    "UNRATE"
]

# Daily datasets
for dataset in daily_datasets:
    
    # Import dataset
    temp_df = pd.read_csv(f"FRED data/{dataset}.csv")
    
    # Convert to datetime
    temp_df["Date"] = pd.to_datetime(temp_df["observation_date"])
    temp_df = temp_df.drop(columns=["observation_date"])
    
    # Merge final dfs
    panel_df = pd.merge(panel_df, temp_df, how="left", on="Date")
    
    
# Monthly datasets
for dataset in monthly_datasets:
    
    # Import dataset
    temp_df = pd.read_csv(f"FRED data/{dataset}.csv")
    
    # Calculate inflation now as it will still be monthly form
    if dataset == "CPIAUCSL":
        temp_df["Inflation rate"] = temp_df["CPIAUCSL"] / temp_df["CPIAUCSL"].shift() - 1
        temp_df = temp_df.drop(columns=["CPIAUCSL"])
    
    # Convert to datetime
    temp_df["Date"] = pd.to_datetime(temp_df["observation_date"])
    temp_df = temp_df.drop(columns=["observation_date"])
    
    # Create separate df with daily dates for merging
    dates = list(panel_df["Date"].unique())
    date_df = pd.DataFrame({"Date": dates})
    date_df['Month'] = date_df['Date'].dt.to_period('M')
    
    # Convert daily datetime to month column
    temp_df["Month"] = temp_df["Date"].dt.to_period("M")
    temp_df = temp_df.drop(columns=["Date"])
    
    # Merge so you have daily data
    temp_df = pd.merge(date_df, temp_df, how="left", on="Month")
    
    # Drop month column
    temp_df = temp_df.drop(columns=["Month"])
    
    # Merge final dfs
    panel_df = pd.merge(panel_df, temp_df, how="left", on="Date")
    

# Change two column names to correct
panel_df = panel_df.rename(columns={
    "TOTAL ASSETS_x": "HISTORIC VOLATILITY",
    "TOTAL ASSETS_y": "TOTAL ASSETS"
    })


#%% Investing.com data (index data)

# Import data CSVs
sp500_index = pd.read_csv("Investing.com data/SP500 index.csv")
spy_etf = pd.read_csv("Investing.com data/SPY ETF.csv")

# Store in a dictionary
index_dfs = {
    "sp500_index": sp500_index,
    "spy_etf": spy_etf
}

# Loop and clean
for name, temp_df in index_dfs.items():
    for col in temp_df.columns:
        
        if col == "Date":
            continue
        
        if temp_df[col].dtype == "object":
            temp_df[col] = temp_df[col].str.replace(",", "", regex=False)
            temp_df[col] = temp_df[col].str.replace("%", "", regex=False)
            temp_df[col] = temp_df[col].str.replace("M", "", regex=False)
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

    # Convert percentage to decimal
    temp_df["Change %"] = temp_df["Change %"] / 100
    
    # Adjust for millions in volume
    temp_df["Vol."] = temp_df["Vol."] * 10**6
    
    # Convert date to datetime
    temp_df["Date"] = pd.to_datetime(temp_df["Date"])

    # Rename columns
    temp_df.rename(columns={"Vol.": "Volume", "Change %": "Return"}, inplace=True)

    # Save back the cleaned DataFrame
    index_dfs[name] = temp_df

# Reassign names
sp500_index = index_dfs["sp500_index"]
spy_etf = index_dfs["spy_etf"]

sp500_index_full = pd.DataFrame({
    "Date": sp500_index["Date"],
    "CloseSP500": sp500_index["Price"],
    "OpenSP500": sp500_index["Open"],
    "HighSP500": sp500_index["High"],
    "LowSP500": sp500_index["Low"],
    "VolumeSPY": spy_etf["Volume"],
    "ReturnSP500": sp500_index["Return"]
    })

panel_df = pd.merge(panel_df, sp500_index_full, how="left", on="Date")


# Finally, filter on non-NaN values for SP500 data points (to keep trading days only)
panel_df = panel_df[panel_df["CloseSP500"].notna()]

# Also drop this one duplicate comum
panel_df = panel_df.drop(columns=["SALES PER SHARE.1"])


#%% Fama-French Five Factor data

# Import data
ff = pd.read_csv("Factor data/Five_factor_daily.csv", skiprows=3)
ff = ff[:-1]

# Rename date column and change to datetime format
ff.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")

# Divide by 100 to get decimals
ff[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']] = ff[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']] / 100

# Merge with panel df
panel_df = pd.merge(panel_df, ff, on="Date", how="left")


#%% Final export

# Export final panel dataframe
panel_df.to_csv("Panel data/sp500_panel.csv")

# Save dtypes in json file as well
dtypes = panel_df.dtypes.astype(str).replace("datetime64[ns]", "object")
dtypes.to_json("Panel data/sp500_panel_dtypes.json")

# Final print
print("Panel dataframe exported successfully")


