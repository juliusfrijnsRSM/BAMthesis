#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:04:58 2025

@author: juliusfrijns
"""

# Script to add variable names to 2 sheets that annoyingly don't have them
# Thx datastream :/

# Imports
import pandas as pd

# Function that adds varnames to columns with only company name in them
def add_varnames(sheet, varname):
    
    for col in sheet.columns:
        
        if col == "Name":
            continue
        
        if " - " not in col:
            sheet = sheet.rename(columns = {col: f"{col} - {varname}"})
            
    return sheet


# SP500 - 1: need to add PRICE

sheet1 = pd.read_excel(
    "Datastream/Datastream batch 1 HC.xlsx",
    sheet_name = "SP500 - 1 HC",
    skiprows = [0, 1, 2, 4],
    engine = "openpyxl"
)

sheet1_adj = add_varnames(sheet1, "PRICE")

print("Finished sheet 1")


# STOXX600 - 1: need to add PRICE

sheet2 = pd.read_excel(
    "Datastream/Datastream batch 2 HC.xlsx",
    sheet_name = "STOXX600 - 1 HC",
    skiprows = [0, 1, 2, 4],
    engine = "openpyxl"
)

sheet2_adj = add_varnames(sheet2, "PRICE")

print("Finished sheet 2")


# SP500 - 3: need to add TOTAL ASSETS

sheet3 = pd.read_excel(
    "Datastream/Datastream batch 1 HC.xlsx",
    sheet_name = "SP500 - 3 HC",
    skiprows = [0, 1, 2, 4],
    engine = "openpyxl"
)

sheet3_adj = add_varnames(sheet3, "TOTAL ASSETS")

print("Finished sheet 3")


# STOXX600 - 3: need to add TOTAL ASSETS

sheet4 = pd.read_excel(
    "Datastream/Datastream batch 2 HC.xlsx",
    sheet_name = "STOXX600 - 3 HC",
    skiprows = [0, 1, 2, 4],
    engine = "openpyxl"
)

sheet4_adj = add_varnames(sheet4, "TOTAL ASSETS")

print("Finished sheet 4")

# Revert nans back to "NA"
sheet1_adj = sheet1_adj.fillna("NA")
sheet2_adj = sheet2_adj.fillna("NA")
sheet3_adj = sheet3_adj.fillna("NA")
sheet4_adj = sheet4_adj.fillna("NA")


# Export to Excel to be copypasted
output_path = "Datastream/Datastream batch varnames added.xlsx"

with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    sheet1_adj.to_excel(writer, sheet_name='SP500 - 1 HC', index=False)
    sheet2_adj.to_excel(writer, sheet_name='STOXX600 - 1 HC', index=False)
    sheet3_adj.to_excel(writer, sheet_name='SP500 - 3 HC', index=False)
    sheet4_adj.to_excel(writer, sheet_name='STOXX600 - 3 HC', index=False)

