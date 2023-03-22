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

def calc_TradeAmt(df):
    exp = (df.loc[df['exp']==1, 'trade_amt_taking'].sum() + df.loc[df['exp']==1,'trade_amt_making'].sum())
    mid = (df.loc[df['mid']==1, 'trade_amt_taking'].sum() + df.loc[df['mid']==1,'trade_amt_making'].sum())
    cheap = (df.loc[df['cheap']==1, 'trade_amt_taking'].sum() + df.loc[df['cheap']==1,'trade_amt_making'].sum())
    return(cheap, mid, exp)

def calc_MakingWt(df):
    exp = df.loc[df['exp']==1,'trade_amt_making'].sum()/(df.loc[df['exp']==1, 'trade_amt_taking'].sum() + df.loc[df['exp']==1,'trade_amt_making'].sum())
    mid = df.loc[df['mid']==1,'trade_amt_making'].sum()/(df.loc[df['mid']==1, 'trade_amt_taking'].sum() + df.loc[df['mid']==1,'trade_amt_making'].sum())
    cheap = df.loc[df['cheap']==1,'trade_amt_making'].sum()/(df.loc[df['cheap']==1, 'trade_amt_taking'].sum() + df.loc[df['cheap']==1,'trade_amt_making'].sum())
    return(cheap, mid, exp)

def calc_OrderWait(df):
    data = df.loc[df['exp']==1]
    exp = np.average(data['making_wait_avg'], weights= data['trade_amt_making'])
    data = df.loc[df['mid']==1]
    mid = np.average(data['making_wait_avg'], weights= data['trade_amt_making'])
    data = df.loc[df['cheap']==1]
    cheap = np.average(data['making_wait_avg'], weights= data['trade_amt_making'])
    return(cheap, mid, exp)

def calc_bbo1(df):
    data = df.loc[(df['exp']==1)&(df['trade_amt_making']>0)]
    exp = data.loc[data['order_price'] > data['prev_bid1p'] ,'trade_amt_making'].sum()/data['trade_amt_making'].sum()
    
    data = df.loc[(df['mid']==1)&(df['trade_amt_making']>0)]
    mid = data.loc[data['order_price'] > data['prev_bid1p'] ,'trade_amt_making'].sum()/data['trade_amt_making'].sum()
    
    data = df.loc[(df['cheap']==1)&(df['trade_amt_making']>0)]
    cheap = data.loc[data['order_price'] > data['prev_bid1p'] ,'trade_amt_making'].sum()/data['trade_amt_making'].sum()
    return(cheap, mid, exp)

def calc_bbo2(df):
    data = df.loc[(df['exp']==1)&(df['trade_amt_making']>0)]
    exp = data.loc[data['order_price'] == data['prev_bid1p'] ,'trade_amt_making'].sum()/data['trade_amt_making'].sum()
    
    data = df.loc[(df['mid']==1)&(df['trade_amt_making']>0)]
    mid = data.loc[data['order_price'] == data['prev_bid1p'] ,'trade_amt_making'].sum()/data['trade_amt_making'].sum()
    
    data = df.loc[(df['cheap']==1)&(df['trade_amt_making']>0)]
    cheap = data.loc[data['order_price'] == data['prev_bid1p'] ,'trade_amt_making'].sum()/data['trade_amt_making'].sum()
    return(cheap, mid, exp)

def calc_bbo3(df):
    data = df.loc[(df['exp']==1)&(df['trade_amt_making']>0)]
    exp = data.loc[data['order_price'] < data['prev_bid1p'] ,'trade_amt_making'].sum()/data['trade_amt_making'].sum()
    
    data = df.loc[(df['mid']==1)&(df['trade_amt_making']>0)]
    mid = data.loc[data['order_price'] < data['prev_bid1p'] ,'trade_amt_making'].sum()/data['trade_amt_making'].sum()
    
    data = df.loc[(df['cheap']==1)&(df['trade_amt_making']>0)]
    cheap = data.loc[data['order_price'] < data['prev_bid1p'] ,'trade_amt_making'].sum()/data['trade_amt_making'].sum()
    return(cheap, mid, exp)


def read_daily(date):
    df = pd.read_parquet("/data/home/jianw/OrderAnalysis/20211115_MakingOrderPortion/concat_data/%s.parquet" %re.sub("[^0-9]", "", date))
    df = df.loc[df['skey']<2300000]
    return (df)
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
    
    df_l = []
    for date in dates:
        print(date)
        df = pd.read_parquet("/data/home/jianw/OrderAnalysis/20211115_MakingOrderPortion/concat_data/%s.parquet" %re.sub("[^0-9]", "", date))
        df_l.append(df)
    df = pd.concat(df_l)
    print(df)
    print('start')
    print('!!!!!!!!!!!!!!! Task1 !!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!! buy !!!!!!!!!!!!!!!!!')
    buy = df.loc[df['order_side']==1]
    buy_top = buy.loc[buy['top']==1]
    print(np.array(calc_TradeAmt(buy))/58)
    print(calc_MakingWt(buy))
    print(np.array(calc_TradeAmt(buy_top))/58)
    print(calc_MakingWt(buy_top))
    
    print('!!!!!!!!!!!!!!! sell !!!!!!!!!!!!!!!!!')
    sell = df.loc[df['order_side']==2]
    sell_buttom = sell.loc[sell['buttom']==1]
    print(np.array(calc_TradeAmt(sell))/58)
    print(calc_MakingWt(sell))
    print(np.array(calc_TradeAmt(sell_buttom))/58)
    print(calc_MakingWt(sell_buttom))
    
    #######################################################
    print('!!!!!!!!!!!!!!! Task2 !!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!! buy !!!!!!!!!!!!!!!!!')
    print(calc_OrderWait(buy))
    print(calc_OrderWait(buy_top))
    print('!!!!!!!!!!!!!!! sell !!!!!!!!!!!!!!!!!')
    print(calc_OrderWait(sell))
    print(calc_OrderWait(sell_buttom))
    
    #######################################################
    print('!!!!!!!!!!!!!!! Task3 !!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!! buy !!!!!!!!!!!!!!!!!')
    print(calc_bbo1(buy))
    print(calc_bbo2(buy))
    print(calc_bbo3(buy))
    print(calc_bbo1(buy_top))
    print(calc_bbo2(buy_top))
    print(calc_bbo3(buy_top))
    
    
    #df.to_parquet('data.parquet')

after = datetime.datetime.now()
print(after - before)