# -*- coding: utf-8 -*-
"""
Created on Sun 2020 June 14
@author: John Merkel

Description: This implements a trained random forest model to predict forward
returns of the crypto-currency BCH

"""

# Import modules
import numpy as np   # linear algebra
import pandas as pd  # data processing
import joblib        # for saving the trained model
import datetime
import matplotlib.pyplot as plt
import requests

# Select style for plots (online suggestion)
plt.style.use('fivethirtyeight')

#%% Define Funcitons #######################################
#
# I found this funtion online.
# This program downloads hourly data. Other functions for minute-by-minute and
# daily data are available (e.g. see crypto_Mo_pro_3.1)

# Import modules
import json

# Download historic hourly price data
def hourly_price_historical(symbol, comparison_symbol, limit, aggregate, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    return df

#%% UPDATE SPREADSHEET DATABASE w/ MOST RECENT DATA for BCH, BTC, ETH, LTC
    
# Read historical data from spreadsheet
excelFile = "cryptoHistPrices.xlsx"

# There are four coins here, but the parameters are optimized for BCH. 
# Functionally, I don't need the three other coins, but they are included
# here for comparison. This code forked off code for an algorithm that 
# traded in all four coins listed.
bchDF = pd.read_excel(excelFile, 'bchHR', index_col=0)
btcDF = pd.read_excel(excelFile, 'btcHR', index_col=0)
ethDF = pd.read_excel(excelFile, 'ethHR', index_col=0)
ltcDF = pd.read_excel(excelFile, 'ltcHR', index_col=0)
#print(ltcDF.tail(5))

# Drop last row, which will not have complete data. E.g. if code is run
# at 8:32 the high, low, close, will be for time period 8:00-8:32 instead 
# of desired period 8:00-9:00.
bchDF.drop(bchDF.tail(1).index, inplace = True)
btcDF.drop(btcDF.tail(1).index, inplace = True)
ethDF.drop(ethDF.tail(1).index, inplace = True)
ltcDF.drop(ltcDF.tail(1).index, inplace = True)
#print(ltcDF.tail(5))

# Get current time to compare, round down to nearest second
import time;
ts = int( time.time() )

# Get time of last data update from spreadsheet.
lastUpdate = bchDF.time.iat[-1]

# Figure out how many hours have passed since last update.
# Add a couple hours as a buffer
hrs = int( (ts - lastUpdate)/3600 ) + 2
print('hrs since last update = ', hrs - 2)

# Fetch data since last update. 
# Bar width (in hours?). Part of the API
time_delta = 1 

# Download Data. Most recent on bottom
bchNewDF = hourly_price_historical('BCH', 'USD', hrs, time_delta)
btcNewDF = hourly_price_historical('BTC', 'USD', hrs, time_delta)
ethNewDF = hourly_price_historical('ETH', 'USD', hrs, time_delta)
ltcNewDF = hourly_price_historical('LTC', 'USD', hrs, time_delta)

# Remove rows with timestamp later than most recently downloaded data 
bchDF = bchDF[ bchDF.time < bchNewDF.time.iat[0] ]
btcDF = btcDF[ btcDF.time < btcNewDF.time.iat[0] ]
ethDF = ethDF[ ethDF.time < ethNewDF.time.iat[0] ]
ltcDF = ltcDF[ ltcDF.time < ltcNewDF.time.iat[0] ]

# Concatenate most recent data to dataframes and reindex.
frames = [bchDF, bchNewDF]
bchDF = pd.concat(frames, sort=True)
bchDF = bchDF.reset_index(drop=True)
frames = [btcDF, btcNewDF]
btcDF = pd.concat(frames, sort=True)
btcDF = btcDF.reset_index(drop=True)
frames = [ethDF, ethNewDF]
ethDF = pd.concat(frames, sort=True)
ethDF = ethDF.reset_index(drop=True)
frames = [ltcDF, ltcNewDF]
ltcDF = pd.concat(frames, sort=True)
ltcDF = ltcDF.reset_index(drop=True)

# Update spreadsheets
writerObj = pd.ExcelWriter(excelFile)

# Write to sheet in excel file
bchDF.to_excel(writerObj,'bchHR')
btcDF.to_excel(writerObj,'btcHR')
ethDF.to_excel(writerObj,'ethHR')
ltcDF.to_excel(writerObj,'ltcHR')

# Write to disk
writerObj.save() 
    
##############################################################
# Create data to input into trained model

# Create time windows (in hrs) to use for regression calculation. We will loop
# thru these and collect the linear regression slopes as data fields.
window_hrs_list = [2,22]

# Number of hours to calculate forecast for. Technically only need two most 
# recent, but it's informative to see how the last several hours have been 
# going. Note that the last forecast will involve data from only a partial hour
numHrs = 6

# Number of rows of data in DF to keep for calculations. 
numRows = numHrs + max(window_hrs_list) + 1

# Make the DF smaller to speed things up. 
pro = bchDF.tail(numRows).copy()

# Create a new column: current 1 hr rtn
pro['rs_1hr'] = pro.close / pro.open

# Calcualte weighted linear regression lines
# import linear regression
from sklearn.linear_model import LinearRegression
wt_lin_reg = LinearRegression()

# Loop through time windows to calculate slopes
for window_hrs in window_hrs_list:
    
    # Weights for Weighted Least Squares. 
    lsWts = np.linspace(1., 3., window_hrs)

    # Calculate linear regression slopes
    # Initialize regression array. Will hold y-int (alpha) and slope (beta)
    regression = np.zeros((len(pro.index), 2))
    for row in range(window_hrs, len(pro.index)):
        x = pro.index[row - window_hrs: row].values.reshape(-1,1)
        y = pro['high'][row - window_hrs: row]
        wt_lin_reg.fit(x, y, sample_weight=lsWts)      
        regression[row] = [wt_lin_reg.coef_, wt_lin_reg.intercept_]
    
    # Place slope (beta) into dataframe
    # Create column label
    col_label = 'slope' + str(window_hrs)
    pro[col_label] = regression[:,0]

# Drop "time" cols
pro = pro.drop(['time', 'timestamp'], axis=1)

# Drop raw prices
pro = pro.drop(['close', 'high', 'low', 'open'], axis=1)

# Drop volumes. (Could be useful. Maybe incorporate later)
pro = pro.drop(['volumefrom', 'volumeto'], axis=1)

# Place columns in correct order. I don't know if this is necessary
pro = pro[['rs_1hr','slope2','slope22']]

#%%
# Read in trained model
filename = 'rand_forest_trained.sav'
clf = joblib.load(filename)

# Make forecasts for last numHrs + 1
pro = pro.tail(numHrs + 1)
pro['forecast'] = clf.predict(pro)
print(pro)

#%%##########################
# Plot BCH 'close, 'high' and 'low' prices. 
# Note regression line corresponds to 'high' prices
numHrs = 20
x = bchDF.tail(numHrs).index.values
y_hi = bchDF.tail(numHrs).high
y_lo = bchDF.tail(numHrs).low
y_close = bchDF.tail(numHrs).close
plt.figure(figsize=(12,6))
plt.plot(x, y_hi, linewidth=1, color='g')
plt.plot(x, y_lo, linewidth=1, color='r')
plt.plot(x, y_close, linewidth=1, color='black')
plt.xticks(rotation='vertical')
plt.show

