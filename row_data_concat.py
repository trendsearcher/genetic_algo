# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:56:54 2019

@author: user_PC
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
delta = timedelta(hours=9)

df1 = pd.read_csv('E:\\history5month\\pureSBER.csv', header= 0, error_bad_lines=False)
df2 = pd.read_csv('E:\\history5month\\pureRTS.csv', header= 0, error_bad_lines=False)
df3 = pd.read_csv('E:\\history5month\\pureSi.csv', header= 0, error_bad_lines=False)

def fix_dates(df):
    '''fix days in dates by 9h gap between ticks'''
    df['Time'] =pd.to_datetime(df.Time)
    df['Timeshift'] = df['Time'].shift(1)
    df['timedelta'] = df['Timeshift'] - df['Time']
    index_next_day_list = df.index[df['timedelta'] > delta].tolist() 
    for i in index_next_day_list:
        df.loc[df.index >= i, ['Time']] += pd.to_timedelta(1, unit='d')
    df = df.drop(['Timeshift', 'timedelta'], 1)
    df.dropna(inplace=True)
    return(df)

def df_datetime_cocat(df1, df2, df3):
    '''concatenates dataframes by datetime column and fill Nans'''
    df1.rename(columns={" <TIME>": "Time", " <VOLUME>": "Volume1", "<PRICE>": "Price1"}, inplace = True)
    df2.rename(columns={" <TIME>": "Time", " <VOLUME>": "Volume2", "<PRICE>": "Price2"}, inplace = True)
    df3.rename(columns={" <TIME>": "Time", " <VOLUME>": "Volume3", "<PRICE>": "Price3"}, inplace = True)
       
    df1 = fix_dates(df1)
    df2 = fix_dates(df2)
    df3 = fix_dates(df3)
    
    sf1 = pd.concat([df1, df2, df3], ignore_index=True, sort=True)
    sf1 = sf1.sort_values(by='Time')
    sf1.fillna(method='pad', inplace=True)
    
    return (sf1)

result = df_datetime_cocat(df1, df2, df3)
result.to_csv('E:\\history5month\\SBER_RTS_Si.csv', index=False)
