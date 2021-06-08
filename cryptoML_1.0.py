# -*- coding: utf-8 -*-
"""
Created on Sun May 3 22:08:55 2020
@author: John Merkel

Description: This is an attempt to use machine learning to predict forward
returns of a crypto coin.

Run crypto_make_data.py first to create the data for this program.
"""

# Import modules
import numpy as np   # linear algebra
import pandas as pd  # data processing

#%%
# Load data
trn = pd.read_csv('cryptoData.csv', index_col = 'Unnamed: 0')

# Drop "time" cols
trn = trn.drop(['time', 'timestamp'], axis=1)

# Drop raw prices
trn = trn.drop(['close', 'high', 'low', 'open'], axis=1)

# Drop volumes. (Could be useful. Maybe incorporate later)
trn = trn.drop(['volumefrom', 'volumeto'], axis=1)

# Use tail as test set. Create tst before modifying trn!
tst = trn.loc[6355:]
trn = trn.loc[:6355]

# Get targets, drop from dataframes
y_trn = trn.rtn_1hr
trn = trn.drop('rtn_1hr', axis=1)
y_tst = tst.rtn_1hr
y_tst = y_tst.to_frame()             # convert to df so we can add col later
tst = tst.drop('rtn_1hr', axis=1)

# Limit inputs
trn = trn[['rs_1hr','slope2','slope22']]
tst = tst[['rs_1hr','slope2','slope22']]

#
# Run this cell to implement XGBoost

import xgboost as xgb

# Max for n_est at 200
# Max for max_depth at 14 
# Max for learning_rate at .042
# Max for subsample at 0.75
# Max for colsample_bytree at 0.10-0.60 (1.658833)
grid = range(1,2, 1) #np.linspace(.5,.7,20)
df = pd.DataFrame(index=range(1,30), columns=grid)
for seed in range(22,23):
    print('seed: ', seed) 
    for param in grid:
        clf = xgb.XGBRegressor(
            objective = 'reg:squarederror',    
            bagging_fraction = 1, # no apparent effect. Value in (0,1]
            n_estimators = 200,
            max_depth = 14,
            learning_rate = .042,
            subsample = 0.75,
            colsample_bytree = .6,   # Default is 1
            colsample_bylevel = 1,   # Default is 1, no change down to 0.5
            colsample_bynode = 1,    # Default is 1, no change down to 0.5
            #num_leaves = param,     # Did not change results
            #missing=-999,
            random_state=seed,
            tree_method='exact'      # Default 'auto'
        )
        
        # Fit classifier
        clf.fit(trn, y_trn)
        
        # Add predictions to targets
        y_tst['forecast'] = clf.predict(tst)
        
        # Calculate Returns        
        y_tst['hold_rtn'] = y_tst.rtn_1hr.cumprod()
        y_tst['model_rtn'] = y_tst.rtn_1hr.where(y_tst.forecast > 1.00, 1).cumprod()
        df.loc[seed,param] = y_tst['model_rtn'].iloc[-1]

#%%
# Load data
trn = pd.read_csv('cryptoData.csv', index_col = 'Unnamed: 0')

# Drop "time" cols
trn = trn.drop(['time', 'timestamp'], axis=1)

# Drop raw prices
trn = trn.drop(['close', 'high', 'low', 'open'], axis=1)

# Drop volumes. (Could be useful. Maybe incorporate later)
trn = trn.drop(['volumefrom', 'volumeto'], axis=1)
#
# Use tail as test set. 
tst = trn.loc[6355:]
trn = trn.loc[:6355]

# Get targets, drop from dataframes
y_trn = trn.rtn_1hr
trn = trn.drop('rtn_1hr', axis=1)
y_tst = tst.rtn_1hr
y_tst = y_tst.to_frame()             # convert to df so we can add col later
tst = tst.drop('rtn_1hr', axis=1)

trn = trn[['rs_1hr','slope2','slope22']]
tst = tst[['rs_1hr','slope2','slope22']]

#
# Run this cell to implement Random Forest
# Specify Model: Random Forest
from sklearn.ensemble import RandomForestRegressor

# GRID SEARCHES
# Optimized parameters for [rs_1hr, slope2, slope22]:
#  max_depth = 16, n_estimators = 512, yields Max = 1.545133
grid = range(19,20)
df = pd.DataFrame(index=range(1,30), columns=grid)
for seed in range(18,19):
    for param in grid:
        clf = RandomForestRegressor(n_estimators = 512,  
                                    max_depth = 16,    
                                    random_state = seed) 
        
        # Fit classifier
        clf.fit(trn, y_trn)
        
        # Add predictions to targets
        y_tst['forecast'] = clf.predict(tst)
        
        # Calculate Returns
        y_tst['hold_rtn'] = y_tst.rtn_1hr.cumprod()
        y_tst['model_rtn'] = y_tst.rtn_1hr.where(y_tst.forecast > 1.00, 1).cumprod()
     #  print('Nest: ', numEst, 'rtn: ', y_tst['model_rtn'].iloc[-1], 
     #         'hold rtn: ', y_tst['hold_rtn'].iloc[-1])
     #   print('seed: ', seed, 'n_est: ', n_est, 'rtn: ', y_tst['model_rtn'].iloc[-1]) 
        df.loc[seed,param] = y_tst['model_rtn'].iloc[-1]

#%%################################################### 
# Test trained model on most recent data

X = pd.read_csv('cryptoOut-hrly-thru-20-6-16.csv', index_col = 'Index')

# Drop "time" cols, raw prices, volumes
X = X.drop(['time', 'timestamp'], axis=1)
X = X.drop(['close', 'high', 'low', 'open'], axis=1)
X = X.drop(['volumefrom', 'volumeto'], axis=1)

# Use tail as test set. Create tst before modifying trn!
X = X.loc[6361:]
Y = X.rtn_1hr
Y = Y.to_frame()             # convert to df so we can add col later
X = X.drop('rtn_1hr', axis=1)

# Limit inputs
X = X[['rs_1hr','slope2','slope22']]

# Add predictions to targets
Y['forecast'] = clf.predict(X)

# Calculate Returns
Y['hold_rtn'] = Y.rtn_1hr.cumprod()
Y['model_rtn'] = Y.rtn_1hr.where(Y.forecast > 1.00, 1).cumprod()

#%% Calculate return for a single column as an algorithm

#Y['slope22'] = X['slope22']
Y['rtn22'] = Y.rtn_1hr.where(X['slope22'] > 0, 1).cumprod()

#%% Calculate returns for each column as an algorithm
for col in tst.columns:
    y_tst[col] = tst[col]
    rtn = y_tst.rtn_1hr.where(y_tst[col] > 0, 1).cumprod()
    print(col, ': ', rtn.iloc[-1])

#%% 
    
for col in trn.columns:
    rtn = y_trn.where(trn[col] > 0, 1).cumprod()
    print(col, ': ', rtn.iloc[-1])

#%%
# Check for missing data

numRows = trn.shape[0]
col_list = trn.columns.values.tolist()

#cols_missing_data = []
for col in col_list:
    missing_ratio = trn[trn[col].isnull()].shape[0] / numRows
    if missing_ratio > 0:
        #cols_missing_data.append(col)
        print(col)
        
#%%
y_tst[['rtn_1hr','hold_rtn','forecast','model_rtn']]

