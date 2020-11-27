# Created May 24, 2020

# This program creates data for use in a cryptocurrency machine learning
# application.

# New to 1.1
#  * Adds window momentum calculations

# Import Modules
import pandas as pd
import numpy as np
#import math
#import os
#import sys
#import datetime

# PARAMETERS
histData = 'cryptoHistPrices-hrly-thru-20-6-16.xlsx'  # Historical data file
excelOut = 'cryptoOut-hrly-thru-20-6-16.xlsx'         # Will hold results
coin = 'BCH'
tab = 'bchHR'     # spreadsheet tab holding data: 'btcHR' 'ethHR' 'ltcHR'

# Read historical data into coin dataframe
coinDF = pd.read_excel(histData, tab, index_col = 0) 
#coinDF = coinDF[3829:5235]

################
# Create time windows (in hrs) to use for momentum calculation. We will loop
# thru these and collect the linear regression slopes as data fields.
max_hrs = 30
window_hrs_list = range(2,max_hrs)

# Create some new columns to use in calculations
# A volitility measure
coinDF['hi_div_lo'] = coinDF.high / coinDF.low
coinDF['hi_div_close'] = coinDF.high / coinDF.close

# Current 1 hr rtn (same as close / close.shift(1))
coinDF['rs_1hr'] = coinDF.close / coinDF.open

# Note nth hr close = n + 1st hr open. So open = close.shift(1)
# Shift by -1 so that current data predicts next hour return
coinDF['rtn_1hr'] = coinDF['rs_1hr'].shift(-1)

# import linear regression
from sklearn.linear_model import LinearRegression
wt_lin_reg = LinearRegression()

# Calculate slopes for weighted linear regression lines
# Loop through time windows
for window_hrs in window_hrs_list:
    
    # Weights for Weighted Least Squares. 
    #lsWts = [math.log(x+2) for x in range(window_hrs)]
    lsWts = np.linspace(1., 3., window_hrs)
    #lsWts = [1]*window_hrs #np.ones(window_hrs, dtype = int) 

    # Calculate linear regression slopes
    # Initialize regression array. Will hold y-int (alpha) and slope (beta)
    regression = np.zeros((len(coinDF.index), 2))
    for row in range(window_hrs, len(coinDF.index)):
        x = coinDF.index[row - window_hrs: row].values.reshape(-1,1)
        y = coinDF['high'][row - window_hrs: row]
        wt_lin_reg.fit(x, y, sample_weight=lsWts)      
        regression[row] = [wt_lin_reg.coef_, wt_lin_reg.intercept_]
    
    # Place slope (beta) into dataframe
    # Create column label
    col_label = 'slope' + str(window_hrs)
    coinDF[col_label] = regression[:,0]

# Calculate momentums using closing prices
# Loop through time windows
for window_hrs in window_hrs_list:
    # Calculate momentum over time frame 
    col_label = 'close_mom' + str(window_hrs)
    coinDF[col_label] = coinDF.close / coinDF.close.shift(window_hrs)    

# Calculate momentums using high prices
# Loop through time windows
for window_hrs in window_hrs_list:
    # Calculate momentum over time frame 
    col_label = 'high_mom' + str(window_hrs)
    coinDF[col_label] = coinDF.high / coinDF.high.shift(window_hrs)    

# Calculate momentums using low prices
# Loop through time windows
for window_hrs in window_hrs_list:
    # Calculate momentum over time frame 
    col_label = 'low_mom' + str(window_hrs)
    coinDF[col_label] = coinDF.low / coinDF.low.shift(window_hrs)    

# Drop rows that are missing data
# Last row does not have next_hr_rtn (target)
coinDF = coinDF[:-1]

# The first max_hrs rows won't have complete data for the slopes of the 
# regression lines, since max_hrs rows are used to calculte the slopes.
coinDF = coinDF[max_hrs - 1:]
 
#%% 
# Write to spreadsheet

writerObj = pd.ExcelWriter(excelOut)
coinDF.to_excel(writerObj, coin)             # writes to an excel sheet
#avgDF.to_excel(writerObj,'avg')
#momDF.to_excel(writerObj,'momentum')
#sortedTkrsDF.to_excel(writerObj,'sorted tkrs')   # writes to an excel sheet
#fwdRtnDF.to_excel(writerObj,'fwdRtn')
writerObj.save()                            # saves the excel workbook to disk

#%% 
# Write to csv file

# Name the index so we can read it in later
#coinDF.index.name = 'idx'
coinDF.to_csv('cryptoOut-hrly-thru-20-6-16.csv')
#%%
# Test Code
from scipy.stats import linregress

days = 14
groups = ['A', 'B', 'C']
data_days = list(range(days)) * len(groups)
values = np.random.rand(days*len(groups))

df = pd.DataFrame(data=zip(sorted(groups*days), data_days, values), 
                  columns=['group', 'day', 'value'])

def get_slope(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope


# calculate slope of regression of last 7 days
days_back = 3

df['rolling_slope'] = df.rolling(window=days_back, min_periods=days_back
                      ).apply(get_slope, raw=False).reset_index(0, drop=True)

#df['rolling_slope'] = df.groupby('group')['value'].rolling(window=days_back,
#                         min_periods=days_back).apply(get_slope, raw=False).reset_index(0, drop=True)

#print(df)

#%%



