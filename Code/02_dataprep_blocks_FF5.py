#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:19:20 2025

@author: juliusfrijns
"""


#%% Data prep for ML analysis

# Imports
import numpy as np
import pandas as pd
import json
from scipy.stats import skew, kurtosis
import statsmodels.api as sm



# Data import
with open("Panel data/sp500_panel_dtypes.json", "r") as f:
    dtypes = json.load(f)
df = pd.read_csv("Panel data/sp500_panel.csv", dtype=dtypes)
df = df.iloc[:, 1:]


# Functions to be used later
def safe_skew(x):
    x = x[~np.isnan(x)]
    return skew(x) if len(x) >= 3 else np.nan

def safe_kurt(x):
    x = x[~np.isnan(x)]
    return kurtosis(x) if len(x) >= 3 else np.nan


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
            if len(batch_df) > 50:
                batches.append(batch_df)
            
            # Reset current batch
            current_batch = []
    
    # Progress tracker
    print(com)
    
    
#%% Relevant variable creation

# Final storage
records = []

# Loop through all batches
for i, batch in enumerate(batches):
    
    # Cap the number of trading days a batch can have (to prevent way to0 long batches)
    if len(batch) > 66:
        batch = batch.iloc[-66:]
    
    # Return and price of stock
    price = batch["PRICE"].values
    log_return = np.log(price[1:] / price[:-1])
    log_return = np.insert(log_return, 0, np.nan)
    #batch["LogReturn"] = log_return
    price_gap = (batch["PRICE HIGH"].iloc[-2] - batch["PRICE LOW"].iloc[-2]) / batch["PRICE"].iloc[-2]
    
    # Abnormal returns
    y = batch["SimpleReturn"][:-1] - batch["RF"][:-1]
    X = sm.add_constant(batch[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]][:-1])
    model = sm.OLS(y, X).fit()
    
    # Predict for this subset
    predicted = model.predict(X)
    residuals = y - predicted
    abnormal_return = residuals.values 
    
    # Return and price of index
    price_sp500 = batch["CloseSP500"].values
    log_return_sp500 = np.log(price_sp500[1:] / price_sp500[:-1])
    log_return_sp500 = np.insert(log_return_sp500, 0, np.nan)
    #batch["LogReturnSP500"] = log_return_sp500
    
    # Risk-free rate
    rf_rate = batch["RF"].values
    
    # Calculate excess return
    excess_log_return = log_return - rf_rate
    #batch["ExcessLogReturn"] = excess_log_return
    
    ## Entity- and time-varying vars
    
    # Abnormal returns on the three days before the earnings call
    abnormal_return_1day_prior = abnormal_return[-2]
    abnormal_return_2day_prior = abnormal_return[-3]
    abnormal_return_3day_prior = abnormal_return[-4]
    
    # Cumulative abnormal returns leading up to earnings announcement
    # Used for to track momentum of stock leading up to earnings call
    three_month_abnormal_return = np.nansum(abnormal_return[:-1])
    one_month_abnormal_return = np.nansum(abnormal_return[-22:-1])
    one_week_abnormal_return = np.nansum(abnormal_return[-6:-1])
    three_day_abnormal_return = np.nansum(abnormal_return[-4:-1])
    
    # Volatility windows (excluding day after earnings day)
    three_month_vol_excess = np.nanstd(excess_log_return[:-1]) * np.sqrt(252)
    three_month_vol = np.nanstd(log_return[:-1]) * np.sqrt(252)
    one_month_vol = np.nanstd(log_return[-22:-1]) * np.sqrt(252)
    one_week_vol = np.nanstd(log_return[-6:-1]) * np.sqrt(252)
    three_day_vol = np.nanstd(log_return[-4:-1]) * np.sqrt(252)
    
    # Skewness windows (excluding day after earnings day)
    three_month_skew = safe_skew(log_return[:-1])
    one_month_skew = safe_skew(log_return[-22:-1])
    one_week_skew = safe_skew(log_return[-6:-1])
    three_day_skew = safe_skew(log_return[-4:-1])
    
    # Kurtosis windows (excluding day after earnings day)
    three_month_kurt = safe_kurt(log_return[:-1])
    one_month_kurt = safe_kurt(log_return[-22:-1])
    one_week_kurt = safe_kurt(log_return[-6:-1])
    three_day_kurt = safe_kurt(log_return[-4:-1])
    
    ## Book value vars and ratios
    # Simple vars
    per = batch["PER"].iloc[-2]
    epr = batch["EARNINGS PER SHR"].iloc[-2]
    div_yield = batch["DIVIDEND YIELD"].iloc[-2]
    turnover_rate = batch["TURNOVER RATE"].iloc[-2]
    
    ptb = batch["PRICE"].mean() / batch["BOOK VALUE PER SHARE"].mean()
    pts = batch["PRICE"].mean() / batch["SALES PER SHARE"].mean()
    
    # Complex vars
    volume = batch["TURNOVER BY VOLUME"]
    
    excess_turnover_volume = (
        (volume.iloc[-2] - volume.iloc[:-2].mean()) / volume.iloc[:-2].mean()
    )
    excess_turnover_volume_3day = (
        (volume.iloc[-4:-1].mean() - volume.iloc[:-4].mean()) / volume.iloc[:-4].mean()
    )
    excess_turnover_volume_5day = (
        (volume.iloc[-6:-1].mean() - volume.iloc[:-6].mean()) / volume.iloc[:-6].mean()
    )
    
    # Ratio vars
    debt_to_assets = batch["TOTAL DEBT"].iloc[-2] / batch["TOTAL ASSETS"].iloc[-2]
    cash_to_assets = batch["CASH"].iloc[-2] / batch["TOTAL ASSETS"].iloc[-2]
    current_to_assets = batch["CURRENT ASSETS"].iloc[-2] / batch["TOTAL ASSETS"].iloc[-2]
    
    # Post-earnings and event returns
    post_earn = excess_log_return[-1]
    event_return = np.nanmean(excess_log_return[-3:])
    hist_vol_daily = three_month_vol_excess / np.sqrt(252)
    
    # Macroeconomic vars
    inflation_rate = batch["Inflation rate"].iloc[-2]
    oil_price_brent = batch["DCOILBRENTEU"].iloc[-2]
    oil_price_wti = batch["DCOILWTICO"].iloc[-2]
    vix = batch["VIXCLS"].iloc[-2]
    high_vix_dummy = vix > 30 # Take 30 as threshold for high VIX regime
    fed_fund_rate = batch["FEDFUNDS"].iloc[-2]
    interest_rate_1mo = batch["REAINTRATREARAT1MO"].iloc[-2]
    interest_rate_1ye = batch["REAINTRATREARAT1YE"].iloc[-2]
    interest_rate_10ye = batch["REAINTRATREARAT1YE"].iloc[-2]
    unemployment_rate = batch["UNRATE"].iloc[-2]
    
    # Index vars
    open_sp500 = batch["OpenSP500"].values
    close_sp500 = batch["CloseSP500"].values
    high_sp500 = batch["HighSP500"].values
    low_sp500 = batch["LowSP500"].values
    
    intra_day_range = (high_sp500[-2] - low_sp500[-2]) / close_sp500[-2]
    
    last_day_index_return = close_sp500[-2]
    
    # Volatility windows for index (excluding earnings day)
    three_month_vol_sp500 = np.nanstd(log_return_sp500[:-1]) * np.sqrt(252)
    one_month_vol_sp500 = np.nanstd(log_return_sp500[-22:-1]) * np.sqrt(252)
    one_week_vol_sp500 = np.nanstd(log_return_sp500[-6:-1]) * np.sqrt(252)
    three_day_vol_sp500 = np.nanstd(log_return_sp500[-4:-1]) * np.sqrt(252)
    
    # Surprise flags
    surprise = int(abs(post_earn) > 1.96 * hist_vol_daily)
    surprise_pos = int(post_earn > 1.96 * hist_vol_daily)
    surprise_neg = int(post_earn < -1.96 * hist_vol_daily)
    surprise_dir = surprise_pos - surprise_neg

    # Date-based features
    dt = pd.to_datetime(batch["Date"].values[-1])
    day_report = dt.day_name()
    year_report = dt.year
    quarter = batch["Quarter"].iloc[0] if "Quarter" in batch.columns else np.nan
    
    # Entity-based features
    geo_location = batch["GeoLocation"].iloc[0]
    industry = batch["Industry"].iloc[0]
    sub_industry = batch["SubIndustry"].iloc[0]

    # Extract company name and ticker
    company = batch["Company"].iloc[0]
    ticker = batch["Ticker"].iloc[0]
    
    # === TECHNICAL INDICATORS ===
    
    # Sharpe Ratio
    sharpe_ratio = np.nanmean(excess_log_return[:-1]) / np.nanstd(excess_log_return[:-1])

    # EMA (12, 26)
    ema_12 = batch['PRICE'].ewm(span=12, adjust=False).mean()
    ema_26 = batch['PRICE'].ewm(span=26, adjust=False).mean()
    EMA12_Last = ema_12.iloc[-2]
    EMA26_Last = ema_26.iloc[-2]
    
    # MACD and Signal
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    MACD = macd.iloc[-2]
    MACD_Signal = macd_signal.iloc[-2]
    
    # ATR (14)
    high = batch["PRICE HIGH"]
    low = batch["PRICE LOW"]
    close = batch["PRICE"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()
    ATR14_Last = atr_14.iloc[-2]
    ATR14_Mean = atr_14.mean()
    
    # OBV
    direction = np.sign(close.diff())
    obv = (direction * batch["TURNOVER BY VOLUME"]).fillna(0).cumsum()
    OBV_Change = obv.iloc[-2] - obv.iloc[0]
    
    # VROC (10)
    vroc_10 = batch["TURNOVER BY VOLUME"].pct_change(periods=10) * 100
    VROC10_Last = vroc_10.iloc[-2]
    VROC10_Mean = vroc_10.mean()
    
    # MFI (14)
    tp = (high + low + close) / 3
    mf = tp * batch["TURNOVER BY VOLUME"]
    pos_flow = mf.where(tp > tp.shift(1), 0)
    neg_flow = mf.where(tp < tp.shift(1), 0)
    pos_sum = pos_flow.rolling(14).sum()
    neg_sum = neg_flow.rolling(14).sum()
    mfi = 100 - (100 / (1 + (pos_sum / neg_sum)))
    MFI14_Last = mfi.iloc[-2]
    MFI14_Mean = mfi.mean()
    
    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    RSI14_Last = rsi.iloc[-2]
    RSI14_Mean = rsi.mean()
    
    # Stochastic Oscillator (%K and %D)
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = 100 * (close - low_14) / (high_14 - low_14)
    stoch_d = stoch_k.rolling(3).mean()
    StochK_Last = stoch_k.iloc[-2]
    StochD_Last = stoch_d.iloc[-2]
    
    # Final weird feature: batch length
    batch_length = len(batch)
    
    
    # FOR SIMULATION ONLY!!!!
    post_abnormal_return = abnormal_return[-1]
    post_log_return = log_return[-1]
    post_sp500_return = log_return_sp500[-1]

    # Build record for final dataframe
    records.append({
        # ID vars
        "Company": company,
        "Ticker": ticker,
        
        # Categorical time vars
        "DayReport": day_report,
        "QuarterReport": quarter,
        "YearReport": year_report,
        
        # Categorical entity vars
        "Country": geo_location,
        "Industry": industry,
        "SubIndustry": sub_industry,
        
        # Excess returns 3 days prior
        "AbnormalReturn1DayPrior": abnormal_return_1day_prior * 100,
        "AbnormalReturn2DayPrior": abnormal_return_2day_prior * 100,
        "AbnormalReturn3DayPrior": abnormal_return_3day_prior * 100,
        
        # Excess return summed (momentum) vars
        "ThreeMonthAbnCumRet": three_month_abnormal_return * 100,
        "OneMonthAbnCumRet": one_month_abnormal_return * 100,
        "OneWeekAbnCumRet": one_week_abnormal_return * 100,
        "ThreeDayAbnCumRet": three_day_abnormal_return * 100,
        
        # Index vars
        "LastDayIndexReturn": last_day_index_return * 100,
        "IntraDayRangeIndex": intra_day_range,
        "ThreeMonthVolatilityIndex": three_month_vol_sp500,
        "OneMonthVolatilityIndex": one_month_vol_sp500,
        "OneWeekVolatilityIndex": one_week_vol_sp500,
        "ThreeDayVolatilityIndex": three_day_vol_sp500,
        
        # Pre-earnings price gap
        "PriceGap": price_gap,
        
        # Volatility vars
        "ThreeMonthVolatility": three_month_vol,
        "OneMonthVolatility": one_month_vol,
        "OneWeekVolatility": one_week_vol,
        "ThreeDayVolatility": three_day_vol,
        
        # Skewness vars
        "ThreeMonthSkew": three_month_skew,
        "OneMonthSkew": one_month_skew,
        "OneWeekSkew": one_week_skew,
        "ThreeDaySkew": three_day_skew,
        
        # Kurtosis vars
        "ThreeMonthKurt": three_month_kurt,
        "OneMonthKurt": one_month_kurt,
        "OneWeekKurt": one_week_kurt,
        "ThreeDayKurt": three_day_kurt,

        # Simple panel vars
        "PriceEarningsRatio": per,
        "EarningsPerShare": epr,
        "DividendYield": div_yield,
        "TurnoverRate": turnover_rate,
        "PriceToBookRatio": ptb,
        "PriceToSalesRatio": pts,
        
        # Complex panel vars
        "ExcessTurnoverVolume": excess_turnover_volume,
        "ExcessTurnoverVolume3Day": excess_turnover_volume_3day,
        "ExcessTurnoverVolume5Day": excess_turnover_volume_5day,
        
        # Macroeconomic vars
        "InflationRate": inflation_rate,
        "OilPriceBrent": oil_price_brent,
        "OilPriceWTI": oil_price_wti,
        "VIX": vix,
        "HighVIXBinary": high_vix_dummy,
        "FedFundRate": fed_fund_rate,
        "InterestRate1Mo": interest_rate_1mo,
        "InterestRate1Ye": interest_rate_1ye,
        "InterestRate10Ye": interest_rate_10ye,
        "UnemploymentRate": unemployment_rate,
        
        # Technical indicators
        "SharpeRatio": sharpe_ratio,
        "EMA12Last": EMA12_Last,
        "EMA26Last": EMA26_Last,
        "MACD": MACD,
        "MACDSignal": MACD_Signal,
        "ATR14Last": ATR14_Last,
        "ATR14Mean": ATR14_Mean,
        "OBVChange": OBV_Change,
        "VROC10Last": VROC10_Last,
        "VROC10Mean": VROC10_Mean,
        "MFI14Last": MFI14_Last,
        "MFI14Mean": MFI14_Mean,
        "RSI14Last": RSI14_Last,
        "RSI14Mean": RSI14_Mean,
        "StochKLast": StochK_Last,
        "StochDLast": StochD_Last,
        
        # Unusual variables (decide later if keep or no)
        "BatchLength": batch_length,
        
        # FOR SIMULATION ONLY!!! Needs to be taken out again before analysis!!!
        # Is positioned here to not screw up x and y indexing in analysis
        "AbnormalReturnSimulation": post_abnormal_return * 100,
        "RawReturnSimulation": post_log_return * 100,
        "MarketReturnSimulation": post_sp500_return * 100,
        
        # Surprise (dependent) vars
        "Surprise": surprise,
        "SurprisePos": surprise_pos,
        "SurpriseNeg": surprise_neg,
        "SurpriseDir": surprise_dir
    })

    print(f"{i + 1}/{len(batches)}")

# Final DataFrame
df_batches = pd.DataFrame(records)

# Export final df
df_batches.to_csv("Block data/batches_prepared.csv")

# Save dtypes in json file as well
dtypes = df_batches.dtypes.astype(str).replace("datetime64[ns]", "object")
dtypes.to_json("Block data/batches_dtypes.json")

# Final print
print("Block dataframe exported successfully")


