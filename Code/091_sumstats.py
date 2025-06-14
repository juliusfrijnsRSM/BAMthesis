#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 17:05:51 2025

@author: juliusfrijns
"""

#%% Script that prints summary stats and other results

# Imports
import numpy as np
import pandas as pd
import json

# Data import for summary statistics
with open("Block data/batches_dtypes.json", "r") as f:
    dtypes1 = json.load(f)
df1 = pd.read_csv("Block data/batches_prepared.csv", dtype=dtypes1)
df1 = df1.iloc[:, 1:]

with open("Block data/batches_postearn_dtypes.json", "r") as f:
    dtypes2 = json.load(f)
df2 = pd.read_csv("Block data/batches_postearn.csv", dtype=dtypes2)
df2 = df2.iloc[:, 1:]

# Drop NaN obs
df1 = df1.dropna()
df2 = df2[df2["Surprise"] == 1].dropna()

# Select relevant columns
cols1 = [
        "AbnormalReturn1DayPrior",
        "AbnormalReturn2DayPrior",
        "AbnormalReturn3DayPrior",
        "ThreeMonthAbnCumRet",
        "OneMonthAbnCumRet",
        "OneWeekAbnCumRet",
        "ThreeDayAbnCumRet",'PriceGap', 'ThreeMonthVolatility',
    'OneMonthVolatility', 'OneWeekVolatility', 'ThreeDayVolatility',
    'ThreeMonthSkew', 'OneMonthSkew', 'OneWeekSkew', 'ThreeDaySkew',
    'ThreeMonthKurt', 'OneMonthKurt', 'OneWeekKurt', 'ThreeDayKurt',
    'PriceEarningsRatio', 'EarningsPerShare', 'DividendYield',
    'TurnoverRate', 'PriceToBookRatio', 'PriceToSalesRatio',
    'ExcessTurnoverVolume', 'ExcessTurnoverVolume3Day',
    'ExcessTurnoverVolume5Day',
    'EMA12Last',
    'EMA26Last', 'MACD', 'MACDSignal', 'ATR14Last', 'ATR14Mean',
    'OBVChange', 'MFI14Last', 'MFI14Mean',
    'RSI14Last', 'RSI14Mean', 'StochKLast', 'StochDLast', 'BatchLength',
    'Surprise', 'SurprisePos', 'SurpriseNeg', 'SurpriseDir'
]

cols2 = [
        "AbnormalReturn1DayPrior",
        "AbnormalReturn2DayPrior",
        "AbnormalReturn3DayPrior",
        "ThreeMonthAbnCumRet",
        "OneMonthAbnCumRet",
        "OneWeekAbnCumRet",
        "ThreeDayAbnCumRet",'PriceGap', 'ThreeMonthVolatility',
    'OneMonthVolatility', 'OneWeekVolatility', 'ThreeDayVolatility',
    'ThreeMonthSkew', 'OneMonthSkew', 'OneWeekSkew', 'ThreeDaySkew',
    'ThreeMonthKurt', 'OneMonthKurt', 'OneWeekKurt', 'ThreeDayKurt',
    'PriceEarningsRatio', 'EarningsPerShare', 'DividendYield',
    'TurnoverRate', 'PriceToBookRatio', 'PriceToSalesRatio',
    'ExcessTurnoverVolume', 'ExcessTurnoverVolume3Day',
    'ExcessTurnoverVolume5Day',
    'EMA12Last',
    'EMA26Last', 'MACD', 'MACDSignal', 'ATR14Last', 'ATR14Mean',
    'OBVChange', 'MFI14Last', 'MFI14Mean',
    'RSI14Last', 'RSI14Mean', 'StochKLast', 'StochDLast', 'BatchLength',
    'PostAbnormalReturn2Months'
]

# Summary stats
df1_sumstats = df1[cols1].describe().T
df2_sumstats = df2[cols2].describe().T

# Print to latex to be copy pasted into overleaf doc
sumstats1_ltready = df1_sumstats.map(lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, float) else x)
sumstats2_ltready = df2_sumstats.map(lambda x: f"{x:.2f}".rstrip('0').rstrip('.') if isinstance(x, float) else x)

print(sumstats1_ltready.to_latex())
print(sumstats2_ltready.to_latex())


