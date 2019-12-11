# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:14:28 2019

@author: user_PC
"""

from datetime import datetime
import numpy as np
import pandas as pd

input_file_name = 'F:\\skleyka\\SBRF1219.csv'
output_pure_file_name = 'F:\\skleyka\\pure_SBRF1219.csv'

#do not del spaces
colnames = ['<DATE>',' <TIME>',' <BID>',' <ASK>',' <LAST>',' <VOLUME>']
data = pd.read_csv(input_file_name, names=colnames, sep = '\t', header = 0, skip_blank_lines=True)
data.fillna(0, inplace=True)

start_of_session = datetime.strptime('10:00:00.000', '%H:%M:%S.%f')
end_of_session = datetime.strptime('23:50:00.000', '%H:%M:%S.%f')

out_of_session_indexes = []
list_of_local_deal_time_indexes_to_delite = []
sum_of_volume__of_local_deal = 0
index_of_last_part_of_local_deal_list = []
list_of_all_deals_time = []
list_of_prices = []

def del_night(data):
    '''функция удаляет все тики за рамками официального 
       закрытия и открытия биржи. В этих промежутках встречаются артефакты.
       А также она собирает цену из 3 полей.'''
    time_i = 0
    for i, row in data.iterrows(): 
        bid_i = row[' <BID>']
        ask_i = row[' <ASK>']
        last_i = row[' <LAST>']
        average_price = []
        if bid_i != 0:
            average_price.append(bid_i)
        if ask_i != 0:
            average_price.append(ask_i)
        if last_i != 0:
            average_price.append(last_i)
        list_of_prices.append(np.mean(average_price))
        time_i = datetime.strptime(row[' <TIME>'], '%H:%M:%S.%f') 
        if time_i > end_of_session or time_i < start_of_session:
            out_of_session_indexes.append(i)
            
    data['<PRICE>'] = pd.Series(list_of_prices, index=data.index)
    data.drop(columns=['<DATE>', ' <BID>', ' <ASK>', ' <LAST>'], inplace=True)
    data.drop(data.index[out_of_session_indexes], inplace = True)
    return(data)
    
data = del_night(data)    
data['<PRICE>'].fillna(method = 'ffill', inplace=True)
data.to_csv(output_pure_file_name, index=False) 
print('1st stage done')






