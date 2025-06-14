#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 21:45:19 2025

@author: juliusfrijns
"""

# Imports
import pandas as pd
import os
import json

# Static vars
SHEETS_NO = 16
EXPORT_DIR = "Datastream CSVs"

# Ensure output directory exists
os.makedirs(EXPORT_DIR, exist_ok=True)

# SP500 sheets
for sheet_num in range(1, SHEETS_NO + 1):
    
    # Skip extra row in sheet 14 and 15
    skiprows = [0, 1, 2, 4, 5] if (sheet_num == 14 or sheet_num == 15) else [0, 1, 2, 4]
    
    df = pd.read_excel(
        "Datastream/Datastream FULL.xlsx",
        sheet_name=f"SP500 - {sheet_num} HC",
        skiprows=skiprows,
        engine="openpyxl"
    )
    
    # Rename Name column to Date
    df.rename(columns = {"Name": "Date"}, inplace=True)

    # Export CSV
    csv_path = f"{EXPORT_DIR}/SP500 - {sheet_num}.csv"
    df.to_csv(csv_path, index=False)

    # Export dtypes JSON to same directory and replace datetime with object
    dtype_path = f"{EXPORT_DIR}/SP500 - {sheet_num}_dtypes.json"
    dtypes = df.dtypes.astype(str).replace("datetime64[ns]", "object")
    dtypes.to_json(dtype_path)

    print(f"SP500 sheet #{sheet_num} exported")

# STOXX600 sheets
for sheet_num in range(1, SHEETS_NO + 1):
    
    # Skip extra row in sheet 14 and 15
    skiprows = [0, 1, 2, 4, 5] if (sheet_num == 14 or sheet_num == 15) else [0, 1, 2, 4]
    
    df = pd.read_excel(
        "Datastream/Datastream FULL.xlsx",
        sheet_name=f"STOXX600 - {sheet_num} HC",
        skiprows=[0, 1, 2, 4],
        engine="openpyxl"
    )

    # Export CSV
    csv_path = f"{EXPORT_DIR}/STOXX600 - {sheet_num}.csv"
    df.to_csv(csv_path, index=False)

    # Export dtypes JSON to same directory and replace datetime with object
    dtype_path = f"{EXPORT_DIR}/STOXX600 - {sheet_num}_dtypes.json"
    dtypes = df.dtypes.astype(str).replace("datetime64[ns]", "object")
    dtypes.to_json(dtype_path)

    print(f"STOXX600 sheet #{sheet_num} exported")

    
