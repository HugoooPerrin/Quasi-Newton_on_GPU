#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""
Feature engineering for the Solar Wind dataset
"""



#=========================================================================================================
#=========================================================================================================
#================================ 0. MODULES


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from lightgbm import LGBMClassifier


#=========================================================================================================
#=========================================================================================================
#================================ 1. FUNCTIONS


#================================
# MEAN

def compute_rolling_mean(data, features, time_windows, center=False):

    data_new = data.copy()

    for feature in features:
        for time_window in time_windows:

            name = '_'.join([feature, time_window, 'mean'])
            data_new[name] = data_new[feature].rolling(time_window, center=center).mean().ffill().bfill()
            data_new[name].astype(data_new[feature].dtype)

    return data_new

def compute_inverse_rolling_mean(data, features, time_windows, center=False):

    data_new = data.iloc[::-1].set_index(data.index).copy()

    for feature in features:
        for time_window in time_windows:

            name = '_'.join([feature, time_window, 'reverse_mean'])
            data_new[name] = data_new[feature].rolling(time_window, center=center).mean().ffill().bfill()
            data_new[name].astype(data_new[feature].dtype)

    return data_new.iloc[::-1].set_index(data.index)

#================================
# STD

def compute_rolling_std(data, features, time_windows, center=False):

    data_new = data.copy()

    for feature in features:
        for time_window in time_windows:

            name = '_'.join([feature, time_window, 'std'])
            data_new[name] = data_new[feature].rolling(time_window, center=center).std().ffill().bfill()
            data_new[name].astype(data_new[feature].dtype)

    return data_new

def compute_inverse_rolling_std(data, features, time_windows, center=False):

    data_new = data.iloc[::-1].set_index(data.index).copy()

    for feature in features:
        for time_window in time_windows:

            name = '_'.join([feature, time_window, 'reverse_std'])
            data_new[name] = data_new[feature].rolling(time_window, center=center).std().ffill().bfill()
            data_new[name].astype(data_new[feature].dtype)

    return data_new.iloc[::-1].set_index(data.index)

#================================
# MEDIAN

def compute_rolling_median(data, features, time_windows, center=False):

    data_new = data.copy()

    for feature in features:
        for time_window in time_windows:

            name = '_'.join([feature, time_window, 'median'])
            data_new[name] = data_new[feature].rolling(time_window, center=center).median().ffill().bfill()
            data_new[name].astype(data_new[feature].dtype)

    return data_new

def compute_inverse_rolling_median(data, features, time_windows, center=False):

    data_new = data.iloc[::-1].set_index(data.index).copy()

    for feature in features:
        for time_window in time_windows:

            name = '_'.join([feature, time_window, 'reverse_median'])
            data_new[name] = data_new[feature].rolling(time_window, center=center).median().ffill().bfill()
            data_new[name].astype(data_new[feature].dtype)

    return data_new.iloc[::-1].set_index(data.index)

#================================
# MIN

def compute_rolling_min(data, features, time_windows, center=False):

    data_new = data.copy()

    for feature in features:
        for time_window in time_windows:

            name = '_'.join([feature, time_window, 'min'])
            data_new[name] = data_new[feature].rolling(time_window, center=center).min().ffill().bfill()
            data_new[name].astype(data_new[feature].dtype)

    return data_new

def compute_inverse_rolling_min(data, features, time_windows, center=False):

    data_new = data.iloc[::-1].set_index(data.index).copy()

    for feature in features:
        for time_window in time_windows:

            name = '_'.join([feature, time_window, 'reverse_min'])
            data_new[name] = data_new[feature].rolling(time_window, center=center).min().ffill().bfill()
            data_new[name].astype(data_new[feature].dtype)

    return data_new.iloc[::-1].set_index(data.index)

#================================
# MAX

def compute_rolling_max(data, features, time_windows, center=False):

    data_new = data.copy()

    for feature in features:
        for time_window in time_windows:

            name = '_'.join([feature, time_window, 'max'])
            data_new[name] = data_new[feature].rolling(time_window, center=center).max().ffill().bfill()
            data_new[name].astype(data_new[feature].dtype)

    return data_new

def compute_inverse_rolling_max(data, features, time_windows, center=False):

    data_new = data.iloc[::-1].set_index(data.index).copy()

    for feature in features:
        for time_window in time_windows:

            name = '_'.join([feature, time_window, 'reverse_max'])
            data_new[name] = data_new[feature].rolling(time_window, center=center).max().ffill().bfill()
            data_new[name].astype(data_new[feature].dtype)

    return data_new.iloc[::-1].set_index(data.index)


#================================
# VAR SELECTION

def select_best_var(data, y, num):
    
    importance = pd.DataFrame()
    importance['name'] = data.columns

    model = LGBMClassifier(objective='binary', verbose=-1, n_estimators=60, n_jobs=-1)

    model.fit(data, y)
    importance['importance'] = model.feature_importances_

    best_var = importance.sort_values(by=['importance'], ascending=False, inplace=False)
    best_var = best_var['name'][:num]
    best_var = best_var.values

    return best_var



#=========================================================================================================
#=========================================================================================================
#================================ 2. FEATURE EXTRACTOR



class FeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        
        self.time_windows = ['1h', '3h', '5h', '10h', '20h']


    def transform(self, data):

        time_windows = self.time_windows
        features = data.columns

        ##====================================================================
        ## Mean/median
        print('Transforming mean/median', end='...')
        data = compute_rolling_mean(data, features, time_windows)
        data = compute_inverse_rolling_mean(data, features, time_windows)

        data = compute_rolling_median(data, features, time_windows)
        data = compute_inverse_rolling_median(data, features, time_windows)
        print('done')


        ##====================================================================
        ## std
        print('Transforming std', end='...')
        data = compute_rolling_std(data, features, time_windows)
        data = compute_inverse_rolling_std(data, features, time_windows)
        print('done')


        ##====================================================================
        ## Min/max
        print('Transforming min/max', end='...')
        data = compute_rolling_min(data, features, time_windows)
        data = compute_inverse_rolling_min(data, features, time_windows)

        data = compute_rolling_max(data, features, time_windows)
        data = compute_inverse_rolling_max(data, features, time_windows)
        print('done')
        

        return data



#=========================================================================================================
#=========================================================================================================
#================================ 3. MAIN


if __name__ == '__main__':

    print('>> Loading data')
    X = pd.read_parquet('data/data_train.parquet')

    print('>> Feature engineering')
    feature_extractor = FeatureExtractor()
    X = feature_extractor.transform(X)

    X = np.array(X, dtype=np.float32)

    np.save(file='data/data_train.npy', arr=X)