#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:10:38 2025

@author: juliusfrijns
"""

# Import packages
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read and print the stock tickers that make up S&P500
ticker = "ASML.AS"

# Get the data for this ticker from yahoo finance

data = yf.download(
    ticker,
    period = '6mo', 
    auto_adjust=True
    )['Close']

# Assume 'data' is a pandas DataFrame with a datetime index and one column of values
data_subset = data.loc['2024-10-16':]  # Select data from the given date onward

# Convert datetime index to numerical values for regression
X = np.array((data_subset.index - data_subset.index[0]).days).reshape(-1, 1)  # Days since start
y = data_subset.values  # Corresponding values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict trendline values
X_full = np.array((data_subset.index - data_subset.index[0]).days).reshape(-1, 1)
trendline = model.predict(X_full)

# Plot original data
plt.plot(data, label="Close")
plt.title(ticker)

# Add vertical line at October 16, 2024
plt.axvline(pd.Timestamp('2024-10-16'), color='r', linestyle='--', linewidth=1, label="Earnings call")

# Plot trendline
plt.plot(data_subset.index, trendline, color='black', linestyle='-', linewidth=2, label="Trendline")

plt.legend()
plt.show()