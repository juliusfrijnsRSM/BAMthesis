#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 19:00:31 2025

@author: juliusfrijns
"""

# Transform date column

# Create a copy of imported sheet
df = sp500_1.copy()

# Rename date column and transform to datetime format
df.rename(columns = {"Name": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], origin="1899-12-30", unit="D")

# Melt the dataframe to get better format for panel data (tbc)
df_long = df.melt(id_vars="Date", var_name="Raw_Column", value_name="Value")

# Separate the variable names from the company names and extract hyphen "-"
df_long[['Ticker', 'Variable']] = df_long['Raw_Column'].str.extract(r'^(.*)\s*-\s*(.+)$')

# Replace NaNs with company name and Price variable depending on column
df_long['Ticker'] = df_long['Ticker'].fillna(df_long['Raw_Column'].str.strip())
df_long['Variable'] = df_long['Variable'].fillna('PRICE')

# Create new panel df with only relevant columns (for now)
panel_df = df_long[['Date', 'Ticker', 'Variable', 'Value']]

# Widen the dataframe to true panel format
panel_wide = panel_df.pivot_table(
    index=['Date', 'Ticker'],
    columns='Variable',
    values='Value'
).reset_index()


#%%


def transform_panel(df):
    
    # Rename date column and transform to datetime format
    df.rename(columns = {"Name": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], origin="1899-12-30", unit="D")
    
    # Melt the dataframe to get better format for panel data (tbc)
    df_long = df.melt(id_vars="Date", var_name="Raw_Column", value_name="Value")
    
    
    
    df_long[["Company", "Variable"]] = df_long["Raw_Column"].str.split(" - ", expand=True)
    
    df_long["Variable"] = np.where(
        df_long["Variable"].isnull(),
        "PRICE",
        df_long["Variable"]
    )
    
    # Create new panel df with only relevant columns (for now)
    panel_df = df_long[['Date', 'Company', 'Variable', 'Value']]
    
    # Widen the dataframe to true panel format
    panel_wide = panel_df.pivot_table(
        index=['Date', 'Company'],
        columns='Variable',
        values='Value'
    ).reset_index()
    
    panel_wide = panel_wide.sort_values(by=["Company", "Date"]).reset_index(drop=True)
    
    return panel_wide


#%%


# Make into function to save memory

def transform_panel(df):
    
    # Rename date column and transform to datetime format
    df.rename(columns = {"Name": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], origin="1899-12-30", unit="D")
    
    # Drop the error column(s)
    number_error_cols = len(df.loc[:, df.columns.str.startswith('#ERROR')].columns)
    print(f"Number of #ERROR columns: {number_error_cols}")
    df = df.loc[:, ~df.columns.str.startswith('#ERROR')]
    
    # Melt the dataframe to get better format for panel data (tbc)
    df_long = df.melt(id_vars="Date", var_name="Raw_Column", value_name="Value")
    
    # Split the raw column names into company and variable
    df_long[["Company", "Variable"]] = df_long["Raw_Column"].str.rsplit(" - ", n=1, expand=True)
    
    df_long["Variable"] = np.where(
        df_long["Variable"].isnull(),
        "PRICE",
        df_long["Variable"]
    )
    
    # Create new panel df with only relevant columns (for now)
    panel_df = df_long[['Date', 'Company', 'Variable', 'Value']]
    
    # Widen the dataframe to true panel format
    panel_wide = panel_df.pivot_table(
        index=['Date', 'Company'],
        columns='Variable',
        values='Value'
    ).reset_index()
    
    panel_wide = panel_wide.sort_values(by=["Company", "Date"]).reset_index(drop=True)
    
    return panel_wide




# Execute function

allsheets = []
for i in range(4, 5):
    
    print(f"Transforming sheet {i}...")
    df = pd.read_excel(
        "Datastream/Datastream FULL.xlsx",
        sheet_name = f"SP500 - {i} HC",
        skiprows = [0, 1, 2, 4],
        engine = "openpyxl"
    )
    
    allsheets.append(transform_panel(df))
    
    print(f"Sheet {i} finished.\n")
    
    
#%%
    
    
def transform_panel(df):
    
    # Transform date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    
    # Drop the error column(s)
    number_error_cols = len(df.loc[:, df.columns.str.startswith('#ERROR')].columns)
    print(f"Number of #ERROR columns: {number_error_cols}")
    df = df.loc[:, ~df.columns.str.startswith('#ERROR')]
    
    # Convert any columns in object form to datetime to prevent agg error later
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Melt the dataframe to get better format for panel data (tbc)
    df_long = df.melt(id_vars="Date", var_name="Raw_Column", value_name="Value")
    
    # Split the raw column names into company and variable
    df_long[["Company", "Variable"]] = df_long["Raw_Column"].str.rsplit(" - ", n=1, expand=True)
    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
    
    # Create new panel df with only relevant columns (for now)
    panel_df = df_long[['Date', 'Company', 'Variable', 'Value']]
    
    # Widen the dataframe to true panel format
    panel_wide = panel_df.pivot_table(
        index=['Date', 'Company'],
        columns='Variable',
        values='Value'
    ).reset_index()
    
    panel_wide = panel_wide.sort_values(by=["Company", "Date"]).reset_index(drop=True)
    
    return panel_wide


#%%


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:04:09 2025

@author: juliusfrijns

Data prep script for BAM Thesis
"""

# Imports
import numpy as np
import pandas as pd
import json
import warnings

# Constant variables
FIRST_SHEET = 1
LAST_SHEET = 11

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
    11: 3
    }

mapping_sp500 = pd.read_csv("Mappings/SP500_mapping.csv")
company_mapping = mapping_sp500

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
    
    # Zip the mapping df into a dictionary
    mapping_dict = dict(zip(company_mapping["Original"], company_mapping["Mapped To"]))
    
    # Apply the mapping to the long df
    mapped = df_long["Company"].map(mapping_dict)
    df_long["Company"] = mapped.combine_first(df_long["Company"])

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


#%%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:04:09 2025

@author: juliusfrijns

Optimized data prep script for BAM Thesis (full version)
"""

import numpy as np
import pandas as pd
import json
import warnings
import os

# Constants
FIRST_SHEET = 1
LAST_SHEET = 11

num_of_cols = {
    1: 3, 2: 3, 3: 3, 4: 3, 5: 2, 6: 3,
    7: 2, 8: 3, 9: 3, 10: 3, 11: 3
}

# Load company mapping
mapping_sp500 = pd.read_csv("Mappings/SP500_mapping.csv")
mapping_dict = dict(zip(mapping_sp500["Original"], mapping_sp500["Mapped To"]))


def transform_panel_optimized(df, mapping_dict):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Drop error columns
    error_cols = df.columns[df.columns.str.startswith("#ERROR")]
    print(f"Number of #ERROR columns: {len(error_cols)}")
    df.drop(columns=error_cols, inplace=True)

    # Keep only company-variable columns
    keep_cols = [col for col in df.columns if " - " in col]
    df = df[["Date"] + keep_cols]

    # Rename using mapping
    renamed_cols = []
    for col in keep_cols:
        try:
            company, var = col.rsplit(" - ", 1)
        except ValueError:
            company, var = col, ""
        mapped_company = mapping_dict.get(company, company)
        renamed_cols.append(f"{mapped_company} - {var}")
    df.columns = ["Date"] + renamed_cols

    # Chunked melt
    chunks = np.array_split(df.columns[1:], 5)
    melted_parts = []

    for subcols in chunks:
        temp_df = df[["Date"] + list(subcols)]
        temp_melt = temp_df.melt(id_vars="Date", var_name="Raw_Column", value_name="Value")
        temp_melt["Raw_Column"] = temp_melt["Raw_Column"].str.replace("CURRENT ASSETS - TOTAL", "CURRENT ASSETS", regex=False)
        temp_melt[["Company", "Variable"]] = temp_melt["Raw_Column"].str.rsplit(" - ", n=1, expand=True)
        melted_parts.append(temp_melt.drop(columns="Raw_Column"))

    df_long = pd.concat(melted_parts, ignore_index=True)

    # Pivot once
    df_pivot = df_long.pivot_table(index=["Date", "Company"], columns="Variable", values="Value", aggfunc="first").reset_index()

    # Smart type inference (post-pivot)
    for col in df_pivot.columns[2:]:
        sample = df_pivot[col].dropna().astype(str).head(10)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dt_test = pd.to_datetime(sample, errors="coerce")

        if dt_test.notna().mean() > 0.8:
            df_pivot[col] = pd.to_datetime(df_pivot[col], errors="coerce")
            print(f"{col} → converted to datetime")
        else:
            num_test = pd.to_numeric(sample, errors="coerce")
            if num_test.notna().mean() > 0.8:
                df_pivot[col] = pd.to_numeric(df_pivot[col], errors="coerce")
                print(f"{col} → converted to numeric")
            else:
                print(f"{col} → left as object")

    return df_pivot



# Main execution
allsheets = []
for i in range(FIRST_SHEET, LAST_SHEET + 1):
    print(f"Transforming sheet {i}...")

    with open(f"Datastream CSVs/SP500 - {i}_dtypes.json", "r") as f:
        dtypes = json.load(f)

    df = pd.read_csv(f"Datastream CSVs/SP500 - {i}.csv", dtype=dtypes)
    temp_df = transform_panel_optimized(df, mapping_dict)
    temp_df = temp_df.iloc[:, :2 + num_of_cols[i]]
    allsheets.append(temp_df)

    if i == FIRST_SHEET:
        panel_df = temp_df.copy()
    else:
        panel_df = pd.merge(panel_df, temp_df, on=["Date", "Company"], how="left")

    print(f"Sheet {i} finished.\n")

panel_df = panel_df.sort_values(by=["Company", "Date"])


#%%


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:04:09 2025

@author: juliusfrijns

Data prep script for BAM Thesis
"""

# Imports
import numpy as np
import pandas as pd
import json
import warnings

# Constant variables
FIRST_SHEET = 1
LAST_SHEET = 11

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
    11: 3
    }

mapping_sp500 = pd.read_csv("Mappings/SP500_mapping.csv")
company_mapping = mapping_sp500

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
        
    # Zip the mapping df into a dictionary
    mapping_dict = dict(zip(company_mapping["Original"], company_mapping["Mapped To"]))
        
    # Change company names in column
    new_cols = []
    for col in df.columns[1:]:  # Skip 'Date'
    
        if "CURRENT ASSETS - TOTAL" in col:
            col = col.replace("CURRENT ASSETS - TOTAL", "CURRENT ASSETS")
            
        if " - " in col:
            company, var = col.rsplit(" - ", 1)
            mapped_company = mapping_dict.get(company, company)
            new_cols.append(f"{mapped_company} - {var}")
        else:
            new_cols.append(col)
    
    df.columns = [df.columns[0]] + new_cols
    
    # Melt the dataframe to get better format for panel data (tbc)
    df_long = df.melt(id_vars="Date", var_name="Raw_Column", value_name="Value")
    

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





#%% Relevant variable creation

# Columns to keep
cols_to_keep = [
    'DayReport',
    'QuarterReport',
    'ThreeMonthVolatility',
    'OneMonthVolatility',
    'OneWeekVolatility',
    'Surprise',
    'SurprisePos',
    'SurpriseNeg'
    ]


batches_adj = []
batches_neg = []
df_batches = pd.DataFrame(columns=cols_to_keep)

for i, batch in enumerate(batches):
    
    # Datetime
    batch["Datetime"] = batch["Date"].values.astype("datetime64[D]")
    
    # Day of week
    batch["DayOfWeek"] = batch["Datetime"].dt.day_name()
    
    # Day of earnings
    batch["DayReport"] = batch["DayOfWeek"].iloc[-1]
    
    # Quarter of earnings report
    batch["QuarterReport"] = batch["Quarter"].iloc[0]
    
    # Returns
    batch["LogReturn"] = np.log(batch["PRICE"] / batch["PRICE"].shift(1))
    
    # Return volatility
    batch["ThreeMonthVolatility"] = batch["LogReturn"].iloc[:-1].std() * np.sqrt(252)
    batch["OneMonthVolatility"] = batch["LogReturn"].iloc[-22:-1].std() * np.sqrt(252)
    batch["OneWeekVolatility"] = batch["LogReturn"].iloc[-6:-1].std() * np.sqrt(252)
    batch["ThreeDaysVolatility"] = batch["LogReturn"].iloc[-4:-1].std() * np.sqrt(252)
    #batch["OneDayVolatility"] = batch["LogReturn"].iloc[-1].std() * np.sqrt(252)
    
    # Announcement day return
    batch["PostEarningsReturn"] = batch["LogReturn"].iloc[-1]
    
    # Announcement return 3-day window
    batch["EarningsEventReturn"] = batch["LogReturn"].iloc[-3:].mean()
    
    # Earnings surprise dummy
    post_earnings_return = batch["LogReturn"].iloc[-1]
    hist_volatility = batch["ThreeMonthVolatility"].iloc[-1] / np.sqrt(252)
    
    if abs(post_earnings_return) > 1.96 * hist_volatility:
        batch["Surprise"] = 1
        
        if post_earnings_return > 1.96 * hist_volatility:
            batch["SurprisePos"] = 1
            batch["SurpriseNeg"] = 0
            
        elif post_earnings_return < -1.96 * hist_volatility:
            batch["SurprisePos"] = 0
            batch["SurpriseNeg"] = 1
            
            batches_neg.append(batch)
            
        batches_adj.append(batch)
        
    else:
        batch["Surprise"] = 0
        batch["SurprisePos"] = 0
        batch["SurpriseNeg"] = 0
    
    # Append to dataframe
    df_batches = pd.concat([df_batches, batch[cols_to_keep].iloc[-1]], axis=0, ignore_index=True)
    
    # Progress tracker
    print(f"{i+1}/{len(batches)}")


#%%
sm = SMOTE()
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)


# Random forest model

# Train model
model = RandomForestClassifier(
    n_estimators=n_estimators, 
    random_state=RANDOM_STATE
    )
model.fit(X_resampled, y_resampled)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# Gradient boosting model

# Train model
model = XGBClassifier(
    n_estimators=n_estimators, 
    learning_rate=0.05, 
    max_depth=4, 
    random_state=RANDOM_STATE
    )
model.fit(X_resampled, y_resampled)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# MLP model

# Train model
model = MLPClassifier(
    hidden_layer_sizes=(100, 50), 
    activation='relu', 
    solver='adam', 
    max_iter=300,
    random_state=RANDOM_STATE
    )
model.fit(X_resampled, y_resampled)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


#%%

# Train model
model = RandomForestClassifier(
    n_estimators=n_estimators, 
    random_state=RANDOM_STATE
    )
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


#%%

    #excess_turnover_value = (
    #    (val.iloc[-2] - val.iloc[:-2].mean()) / val.iloc[:-2].mean()
    #)
    #excess_turnover_value_3day = (
    #    (val.iloc[-4:-1].mean() - val.iloc[:-4].mean()) / val.iloc[:-4].mean()
    #)
    #excess_turnover_value_5day = (
    #    (val.iloc[-6:-1].mean() - val.iloc[:-6].mean()) / val.iloc[:-6].mean()
    #)
    
    

#%%


    # Volatility windows (excluding earnings day)
    three_month_vol = np.nanstd(log_return[:-1]) * np.sqrt(252)
    one_month_vol = np.nanstd(log_return[-22:-1]) * np.sqrt(252)
    one_week_vol = np.nanstd(log_return[-6:-1]) * np.sqrt(252)
    three_day_vol = np.nanstd(log_return[-4:-1]) * np.sqrt(252)


    # Post-earnings and event returns
    post_earn = log_return[-1]
    event_return = np.nanmean(log_return[-3:])
    hist_vol_daily = three_month_vol / np.sqrt(252)



#%%

# Now we create batches of each earnings period (quarter)

# Empty list for batch dfs to append to
batches = []

# Loop through companies
for com in companies:
    
    # Create mask df and empty list for batch
    mask_df = df[df["Company"] == com].reset_index(drop=True)
    current_batch = []
    
    # Loop through all rows in the mask df
    for i in range(len(mask_df)):
        current_batch.append(mask_df.iloc[i])
        
        # Check for announcement day
        if mask_df.loc[i, "ANNOUNCEMENT_DAY"] == 1:
            
            # Check if there's a next row
            if i + 1 < len(mask_df):
                current_batch.append(mask_df.iloc[i + 1])
            
            # Save the batch
            batch_df = pd.DataFrame(current_batch)
            
            # Append to the list
            if len(batch_df) >= 62:
                
                batch_df = batch_df[-62:]
                
                # Create variables
                batch_df["LogReturn"] = np.log(batch_df["PRICE"] / batch_df["PRICE"].shift())
                batch_df["LogReturnIndex"] = np.log(batch_df["CloseSP500"] / batch_df["CloseSP500"].shift())
                batch_df["ExcessReturn"] = batch_df["LogReturn"] - batch_df["LogReturnIndex"]
                
                batch_df["HiLowGap"] = (batch_df["PRICE HIGH"] - batch_df["PRICE LOW"]) / batch_df["PRICE"]
                batch_df["LogVolume"] = np.log(batch_df["TURNOVER BY VOLUME"])
                
                batch_df["LogVolumeIndex"] = np.log(batch_df["VolumeSPY"])
                batch_df["HiLowGapIndex"] = (batch_df["HighSP500"] - batch_df["LowSP500"]) / batch_df["CloseSP500"]
                
                batch_df["LogBrent"] = np.log(batch_df["DCOILBRENTEU"])
                batch_df["LogWTI"] = np.log(batch_df["DCOILWTICO"])
                batch_df["LogVIX"] = np.log(batch_df["VIXCLS"])
                
                # y var
                hist_vol = np.nanstd(batch_df["ExcessReturn"][:-1])
                
                if batch_df["ExcessReturn"].iloc[-1] > 1.96 * hist_vol:
                    batch_df["Surprise"] = 1
                    
                elif batch_df["ExcessReturn"].iloc[-1] < 1.96 * hist_vol:
                    batch_df["Surprise"] = -1
                    
                else:
                    batch_df["Surprise"] = 0
                    
                relevant_cols = [
                    # Stock vars
                    "ExcessReturn",
                    "HiLowGap",
                    "LogVolume",
                    
                    # Index vars
                    "LogReturnIndex",
                    "HiLowGapIndex",
                    "LogVolumeIndex",
                    
                    # Macro vars
                    "LogBrent",
                    "LogWTI",
                    "LogVIX",
                    "Inflation rate",
                    "FEDFUNDS"
                    "REAINTRATREARAT1MO",
                    "REAINTRATREARAT1YE",
                    "REAINTRATREARAT10Y",
                    "UNRATE"
                ]
                
                batch_relevant = batch_df[relevant_cols]
                    
                batches.append(batch_relevant[-61:])
            
            # Reset current batch
            current_batch = []
    
    # Progress tracker
    print(com)
    
    
    
    
#%%

# Extract classifier and scaler
classifier = best_model.named_steps['clf']
scaler = best_model.named_steps['scaler']

# Original feature names (assuming X_train is a DataFrame)
feature_names = X_train.columns

# Get model coefficients
coefs = classifier.coef_[0]  # Binary classification

# Create importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': coefs
})
importance_df['Abs_Importance'] = importance_df['Importance'].abs()
importance_df = importance_df.sort_values(by='Abs_Importance', ascending=False).head(15)

# Plot
plt.figure(figsize=(8,6))
sns.barplot(x='Abs_Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Top 15 Most Important Features (Logistic Regression)")
plt.tight_layout()
#plt.savefig("Plots/feature_importance/feature_importance_surprise.png")
plt.show()

#%% XGBoost var importance


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

#%% Final model training and simulation evaluation (CatBoost + SMOTE)

# Prepare features and targets
X_train_test = df_train_test.iloc[:, 2:-4].copy()
X_simulation = df_simulation.iloc[:, 2:-4].copy()
y_train_test = df_train_test.iloc[:, -1].copy()
y_simulation = df_simulation.iloc[:, -1].copy()

# Drop YearReport if present
for df_ in [X_train_test, X_simulation]:
    if "YearReport" in df_.columns:
        df_.drop(columns="YearReport", inplace=True)

# Identify string/categorical columns
string_cols = X_train_test.select_dtypes(include=["object", "string"]).columns
for col in string_cols:
    X_train_test[col] = X_train_test[col].astype(str)
    X_simulation[col] = X_simulation[col].astype(str)

# Sample weights
sample_weights = compute_sample_weight(class_weight={0: 3, 1: 1, 2: 3}, y=y_train_test)

# Cat features
cat_features = [X_train_test.columns.get_loc(col) for col in string_cols]

# Remap labels: 0 = neg surprise, 1 = neutral, 2 = pos surprise
y_train_test = y_train_test + 1
y_simulation = y_simulation + 1

# Define pipeline with SMOTE and CatBoost
final_model = Pipeline([
    ("smote", SMOTE(random_state=678329)),
    ("clf", CatBoostClassifier(
        verbose=0,
        random_state=678329,
        loss_function='MultiClass'
    ))
])

# Fit model
final_model.fit(X_train_test, y_train_test, sample_weight=sample_weights, cat_features=cat_features)

# Predict on simulation set
y_sim_pred = final_model.predict(X_simulation)

# Evaluate predictions
print("Classification Report on Simulation Set (2024):")
print(classification_report(y_simulation, y_sim_pred, labels=[0, 1, 2], zero_division=0))


#%%

returns_sim["PerfectAbnormalReturn"] = returns_sim["SurpriseDir"] * returns_sim["AbnormalReturnSimulation"]
returns_sim["PerfectRawReturn"] = returns_sim["SurpriseDir"] * returns_sim["RawReturnSimulation"]

# Abnormal return stats
abnormal_return_mean = returns_sim[returns_sim["SurprisePred"] != 0]["RealizedAbnormalReturn"].mean()
abnormal_return_std = returns_sim[returns_sim["SurprisePred"] != 0]["RealizedAbnormalReturn"].std()
abnormal_return_sharpe = abnormal_return_mean / abnormal_return_std
abnormal_return_perfect = returns_sim["PerfectAbnormalReturn"].mean()

# Raw return stats
raw_return_mean = returns_sim[returns_sim["SurprisePred"] != 0]["RealizedRawReturn"].mean()
raw_return_std = returns_sim[returns_sim["SurprisePred"] != 0]["RealizedRawReturn"].std()
raw_return_sharpe = raw_return_mean / raw_return_std
raw_return_perfect = returns_sim["PerfectRawReturn"].mean()


#%%$


# Make returns df 
returns_sim = df_simulation[sim_columns]

returns_sim["PostAbnormalReturnPredicted"] = y_sim_pred

returns_sim["TradeDrift"] = np.where(
    returns_sim["PostAbnormalReturnPredicted"] > 0, 1, 0
    )
returns_sim["TradeReversal"] = np.where(
    returns_sim["PostAbnormalReturnPredicted"] < 0, -1, 0
    )
returns_sim["TradeBalanced"] = returns_sim["TradeDrift"] + returns_sim["TradeReversal"]

# Drift trading
returns_sim["RealizedAbnormalReturnDrift"] = returns_sim["TradeDrift"] * returns_sim["PostAbnormalReturnSimulation"]
returns_sim["RealizedRawReturnDrift"] = returns_sim["TradeDrift"] * returns_sim["PostRawReturnSimulation"]

# Reversal trading
returns_sim["RealizedAbnormalReturnDrift"] = returns_sim["TradeDrift"] * returns_sim["PostAbnormalReturnSimulation"]
returns_sim["RealizedRawReturnDrift"] = returns_sim["TradeDrift"] * returns_sim["PostRawReturnSimulation"]

# Balanced trading
returns_sim["RealizedAbnormalReturnBalanced"] = returns_sim["TradeBalanced"] * returns_sim["PostAbnormalReturnSimulation"]
returns_sim["RealizedRawReturnDrift"] = returns_sim["TradeDrift"] * returns_sim["PostRawReturnSimulation"]

# Summary table of simulation statistics
stats_summary = returns_sim[returns_sim["Trade"] == 1][["RealizedAbnormalReturn", "RealizedRawReturn"]].describe()


# Import market returns
mkt_return = pd.read_csv("Investing.com data/SP500 index.csv")
mkt_return["Year"] = pd.to_numeric(mkt_return["Date"].str[-4:])

mkt_return = mkt_return[mkt_return["Year"] == 2024]
mkt_return["Return"] = pd.to_numeric(mkt_return["Change %"].str.replace("%", ""))

stats_summary["MarketReturn"] = mkt_return["Return"].describe()

stats_summary = stats_summary.T

stats_summary["Sharpe Ratio"] = stats_summary["mean"] / stats_summary["std"]
stats_summary = stats_summary.round(2)
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
#plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/sim_returns_density.png", dpi=500)
plt.show()




#%% OOS analysis

# Set y-variable
y_train_test = df_train_test["PostAbnormalReturn2Months"].copy()
y_simulation = df_simulation["PostAbnormalReturn2Months"].copy()

# Original sets (no encoding or dimensionality reducing)
X_train_test_original = df_train_test.iloc[:, 2:-8].copy()
X_simulation_original = df_simulation.iloc[:, 2:-8].copy()

# Drop YearReport if present
for tt_set in [X_train_test_original, X_simulation_original]:
    if "YearReport" in tt_set.columns:
        tt_set.drop(columns="YearReport", inplace=True)

# Ordinal encoded sets
string_cols = X_train_test_original.select_dtypes(include=["object", "string"]).columns

X_train_test_ordinal = X_train_test_original.copy()
X_simulation_ordinal = X_simulation_original.copy()

if len(string_cols) > 0:
    
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    X_train_test_ordinal[string_cols] = encoder.fit_transform(X_train_test_ordinal[string_cols])
    X_simulation_ordinal[string_cols] = encoder.transform(X_simulation_ordinal[string_cols])

# Get right X variable dataframes
X_train_test = X_train_test_ordinal.copy()
X_simulation = X_simulation_ordinal.copy()

# Train CatBoostRegressor
final_model = CatBoostRegressor(
    **best_models_params[best_mod_str],
    random_state=RANDOM_STATE,
    verbose=0
)
final_model.fit(X_train_test, y_train_test)

y_sim_pred = final_model.predict(X_simulation)

# Compute metrics
mse = mean_squared_error(y_simulation, y_sim_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_simulation, y_sim_pred)
r2 = r2_score(y_simulation, y_sim_pred)

# Print results
print("Best Parameters:", best_models_params[best_mod_str])
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE", mae)
print("R² Score:", r2)


# Table with in-sample and out-of-sample analysis metrics
stats_comparison = pd.DataFrame(best_models_stats[best_mod_str], index=["In-Sample"])
stats_comparison.loc["Out-of-Sample"] = [mse, rmse, mae, r2] # OOS metrics
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
plt.xlabel("Actual 2-Month Post-Negative-Surprise Abnormal Return")
plt.ylabel("Predicted 2-Month Post-Negative-Surprise Abnormal Return")
plt.title("Out-of-Sample Prediction Accuracy (2024)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/regression_simulation_accuracy.png")
plt.show()

#%% Variable importance


# Get best model
best_mod_str = df_stats.index[0]
#################
best_mod_str = 'AdaBoost'
best_model = best_models[best_mod_str]

# Save model
dump(best_model, "Models/best_mod_regression_pos.joblib")

# Feature importances (CatBoost built-in)
if hasattr(best_model, 'feature_importances_'):
    n_features_model = len(best_model.feature_importances_)
    n_features_data = X_test.shape[1]

    if n_features_model == n_features_data:
        importance_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(15)
    else:
        importance_df = pd.DataFrame({
            'Feature': [f"Feature_{i}" for i in range(n_features_model)],
            'Importance': best_model.feature_importances_
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

# SHAP plot

# Create explainer
explainer = shap.Explainer(best_model)

# Compute SHAP values
shap_values = explainer(X_test)

# Global summary plot
shap.summary_plot(shap_values, X_test, show=True)

# Save plot
shap.summary_plot(shap_values, X_test, show=False)
fig = plt.gcf()
fig.tight_layout()
fig.savefig("Plots/feature_importance/feature_importance_shap_regression_pos.png", dpi=500, bbox_inches="tight")
fig.show()
plt.close(fig)