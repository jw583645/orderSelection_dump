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

def concat_stock(date):
    print(date)
    skeys = os.listdir(data_path + re.sub("[^0-9]", "", date) +'/' )
    skeys = [val for val in skeys if not val.endswith(".csv")]
    skeys = set(np.array([i[:7] for i in skeys]).astype(int))

    DF_l = []
    
    for skey in skeys:
        #df = pd.read_csv( '%s/%s/%s.csv.gz'%(data_path, re.sub("[^0-9]", "", date), skey) )
        #print(skey)
        df = pd.read_parquet( '%s/%s/%s.parquet'%(data_path, re.sub("[^0-9]", "", date), skey) )
        df = df.loc[df['order_type']==2]
        #df = df[['skey', 'prev_yHatBuy90','alp_1d', '1d_ret', 'trade_qty', 'trade_price', 'adjMidF90s', 'amountThisUpdate']]
        DF_l.append(df)
    DF = pd.concat(DF_l)
    
    mta_rank = pd.read_csv("/data/home/jianw/sta/ICTopNames/ICTopButtom_%s.csv" %re.sub("[^0-9]", "", date))
    #print(DF)
    #print(mta_rank)
    DF = DF.merge(mta_rank.drop(columns = 'date'), on =  ['skey', 'interval'], how = 'left')
    
    daily_price = pd.read_csv("/data/rch/raw/marketData_TR/dailyData/stock/2021/stock_%s.csv.gz"%date)
    daily_price['skey'] = np.where(daily_price['ID'].str[:2] == 'SH',  '1' + daily_price['ID'].str[2:], '2' + daily_price['ID'].str[2:]).astype(int)
    daily_price = daily_price.loc[daily_price['skey'].astype(int).isin(set(DF['skey']))][['skey','yclose']]
    daily_price['cheap'] = (daily_price['yclose'] <= np.quantile(daily_price['yclose'], 1/3)).astype(int)
    daily_price['mid'] = ((daily_price['yclose'] > np.quantile(daily_price['yclose'], 1/3)) & (daily_price['yclose'] <= np.quantile(daily_price['yclose'], 2/3))).astype(int)
    daily_price['exp'] = (daily_price['yclose'] > np.quantile(daily_price['yclose'], 2/3)).astype(int)
    daily_price.fillna(0, inplace = True)
    
    DF = DF.merge(daily_price.drop(columns = 'yclose'), on = 'skey', how = 'left')
    
    
    #DF['trade_amt'] = DF['trade_qty'] * DF['trade_price']
    #DF['trade_amt'] = DF['amountThisUpdate']
    #DF['90s_ret'] = (DF['adjMidF90s'] - DF['trade_price']) / DF['trade_price']   
    df = DF.reset_index(drop = True)
    df[['skey', 'date', 'ApplSeqNum', 'time' ,'clockAtArrival', 'bid1p', 'ask1p', 'prev_bid1p', 'prev_ask1p', 'order_side', 'order_price',	'order_qty', 'trade_amt_making',	'trade_amt_taking',	'making_wait_avg',	'taking_wait_avg', 'top', 'buttom', 'cheap', 'mid', 'exp']].to_parquet('/data/home/jianw/OrderAnalysis/20211115_MakingOrderPortion/concat_data/%s.parquet'%(re.sub("[^0-9]", "", date)), index = False)
    print(df)
    #return(df)

before = datetime.datetime.now()          
if __name__ == '__main__':

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/sta/data_jian/'
    sdate = '2021-01-01'
    edate = '2021-03-31'

    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    P = mp.Pool(96)
    P.map(concat_stock, dates)
    P.close()
    P.join()

after = datetime.datetime.now()
print(after - before)