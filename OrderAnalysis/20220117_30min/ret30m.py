import numpy as np
import pandas as pd
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import os, sys
from datetime import date
from datetime import datetime
import datetime
os.environ["OMP_NUM_THREADS"] = '1'
import os
import sys 
import numpy as np
import pandas as pd 
from warnings import filterwarnings 
filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = '1'
import multiprocessing as mp
from datetime import date
from datetime import datetime
#import datetime

# DFS 
import jdfs
from AShareReader import AShareReader

def merge(pair):
    date, skey = pair
    mbd = pd.read_parquet("/data/home/jianw/monetization_research/rawData_jian/data_jian_order_trade_mbd/mbd/%s/%s.parquet" %(date, skey))
    order = pd.read_parquet("/data/home/jianw/monetization_research/rawData_jian/data_jian_order_trade_mbd/order/%s/%s.parquet" %(date, skey))
    trade = pd.read_parquet("/data/home/jianw/monetization_research/rawData_jian/data_jian_order_trade_mbd/trade/%s/%s.parquet" %(date, skey))
    df = pd.read_parquet("/data/home/jianw/monetization_research/rawData_jian/data_jian_20211213/%s/%s.parquet"%(date, skey))
    #print(df)
    

    mbd = mbd[(mbd['time'] >= 93000000000) & (mbd['time'] <= 145655000000)]    
    mbd = pd.merge(mbd, trade[['ApplSeqNum', 'localClockAtArrival']].rename(columns={'localClockAtArrival':'trade_localClockAtArrival'}), how='left', on='ApplSeqNum')
    mbd = pd.merge(mbd, order[['ApplSeqNum', 'localClockAtArrival']].rename(columns={'localClockAtArrival':'order_localClockAtArrival'}), how='left', on='ApplSeqNum')
    mbd['sortingLocalClockAtArrival'] = (mbd['order_localClockAtArrival'].fillna(0) + mbd['trade_localClockAtArrival'].fillna(0)).replace(0, np.nan)
    order = order[order['localClockAtArrival'] >= mbd['sortingLocalClockAtArrival'].iloc[0] + 2500].copy()
    order['sortingLocalClockAtArrival'] = order['localClockAtArrival'] + 1800000000
    order['tag'] = 'order'
    mbd['tag'] = 'zmbd'
    order_mbd = order.append(mbd).sort_values(['sortingLocalClockAtArrival', 'tag'])
    for c in ['bid1p', 'ask1p', 'bid5p', 'ask5p', 'bid1q', 'ask1q', 'bid5q', 'ask5q']:
        order_mbd[c] = order_mbd[c].fillna(method='ffill')
        
            
    order_mbd = order_mbd[order_mbd['order_type'].notnull()][list(order.columns) + ['bid1p', 'ask1p', 'bid1q', 'ask1q','bid5p', 'ask5p', 'bid5q', 'ask5q']]
    order_mbd = order_mbd[['ApplSeqNum','bid1p', 'ask1p', 'bid1q', 'ask1q']].rename(columns = 
                            {'bid1p':'bid1p_30m', 'ask1p':'ask1p_30m', 'bid1q':'bid1q_30m', 'ask1q':'ask1q_30m'})
                            
                            

    df = df.merge(order_mbd, on = 'ApplSeqNum', how = 'left')
    
    
    try:
        os.makedirs('/data/home/jianw/monetization_research/rawData_jian/data_jian_20220119/%s' %(date))
    except:
        pass
    savePath = '/data/home/jianw/monetization_research/rawData_jian/data_jian_20220119/%s/%s.parquet' %(date, skey)
    df.to_parquet(savePath) 



pairs = []
daily_data = reader.Read_Stock_Daily('com_md_eq_cn', 'mdbar1d_jq', startDate, endDate, univ=['IC'])
print(daily_data['skey'] >= 2000000)
daily_data = daily_data.loc[daily_data['skey'] >= 2000000]
print(daily_data)
for i in range(len(daily_data)):
    pairs.append((daily_data.iloc[i]['date'], daily_data.iloc[i]['skey']))
    #generateSingleAggData((20210104, 2000009, saveRoot))
    
    
P = mp.Pool(96)
P.map(loadTrade, pairs)
P.close()
P.join()

#print(merge(20210423, 2000009))