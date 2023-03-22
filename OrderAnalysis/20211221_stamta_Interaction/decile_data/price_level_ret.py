import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import os, sys
import lightgbm as lgbm
import math
from datetime import date
from datetime import datetime
import datetime
from sklearn import linear_model
import statsmodels.api as sm
os.environ["OMP_NUM_THREADS"] = '1'
pd.set_option("display.max_columns", None)


price_level = [0,5,10,20,50, 100, 10000]
price_level_t = ['<5', '5~10', '10~20', '20~50', '50~100', '>100']


def calcRet(date):
    print(date)
    df = pd.read_csv("/data/rch/raw/marketData_TR/dailyData/stock/%s/stock_%s.csv.gz" %(date[0:4], date))
    ic = pd.read_csv("/data/rch/raw/secData_TR/index_member/member_table_IC_%s.csv.gz" %date)
    df = df.loc[(df['ID'].isin(ic['ID'])) & (df['ID'].str[0:2] == 'SZ')]
    '''
    for i in range(6):
        df_local = df.loc[(df['yclose'] > price_level[i]) & (df['yclose'] <=price_level[i+1])]
    '''
    return(df)


if __name__ == '__main__':
    
    order_cap = 1000000
    order_side = 1

    inSample = 1

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/sta/data_jian_newagg/'
    
    sdate = '2021-01-01'
    edate = '2021-06-30'
    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    #dates = date_list.to_list()
    if inSample == 1:
        IS = pd.read_csv("/data/home/jianw/L4PO/data/MTA_IS_Dates_CHN.csv")
        IS.columns = ['date']
        date_list = date_list.loc[date_list.isin(IS['date'])]
    dates = date_list.to_list()
    n = len(dates)
    print(dates)
    
    df_l = []
    for date in dates:
        df_l.append(calcRet(date))
    
    df = pd.concat(df_l)
    
    df.to_csv('/data/home/jianw/OrderAnalysis/20211221_stamta_Interaction/decile_data/price_level_ret.csv', index = False)
    
    
    TradeAmt = []
    Ret = []
    
    for i in range(6):
        df_local = df.loc[(df['yclose'] > price_level[i]) & (df['yclose'] <=price_level[i+1])]
        trade_amt = df_local.amount.sum()/n
        ret = (df_local['dayReturn'] * df_local['amount'] ).sum() / (df_local['amount'] ).sum() * 10000
        
        TradeAmt.append(trade_amt)
        Ret.append(ret)

    df = pd.DataFrame({'price_level': price_level_t, 'tradeAmt': TradeAmt, 'rets (bps)' : Ret })
    df['tradeAmt_wt'] = df['tradeAmt'] / df['tradeAmt'].sum() 
        
        
    #df.to_csv('/data/home/jianw/OrderAnalysis/20211221_stamta_Interaction/decile_data/price_level_ret.csv', index = False)
    #calcRet('2021-04-23')