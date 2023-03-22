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

def stamta(tuple):
    date, order_side, order_cap  = tuple
    print(date)
    try:
        df = pd.read_parquet('/data/home/jianw/OrderAnalysis/20211125_TakingBaseline/concat_data/%s.parquet' %(date.replace('-', '')))
    except:
        print('%s data missing'%date)
        return None
        
    df = df.loc[~df['alpha'].isna(),].copy()
    df['order_amt'] = np.minimum(df['order_amt'], order_cap)
    df['trade_amt'] = np.minimum(df['trade_qty_taking']*df['trade_price'], order_cap)
    df_side = df.loc[df['order_side'] == order_side]
    
    df_side['order_amt_all'] = df_side.groupby('interval')['order_amt'].transform('sum')
    
    if order_side == 1:
        df_side = df_side.sort_values(by = ['interval','alp_1d', 'clockAtArrival'], ascending=[True, True, True]) 
    if order_side == 2:
        df_side = df_side.sort_values(by = ['interval','alp_1d', 'clockAtArrival'], ascending=[True, False, True]) 
    df_side['order_amt_cumsum_mta'] = df_side.groupby('interval')['order_amt'].transform('cumsum')
    df_side['mta_quantile'] = (np.ceil(df_side['order_amt_cumsum_mta'] / df_side['order_amt_all'] * 10)).astype(int)
    
    if order_side == 1:
        df_side = df_side.sort_values(by = ['interval','prev_yHatBuy90', 'clockAtArrival'], ascending=[True, True, True])    
    if order_side == 2:
        df_side = df_side.sort_values(by = ['interval','prev_yHatSell90', 'clockAtArrival'], ascending=[True, True, True])  
    
    df_side['order_amt_cumsum_sta'] = df_side.groupby('interval')['order_amt'].transform('cumsum')
    df_side['sta_quantile'] = (np.ceil(df_side['order_amt_cumsum_sta'] / df_side['order_amt_all'] * 10)).astype(int)
    
    return(df_side)
    

def summary(df_side):
    tradeAmt = []
    ret_1d = []
    mta_q = []
    sta_q = []
    R2 = []
    MTA_coef = []
    STA_coef = []
    for i in range(10):
        for j in range(10):
            mta_q.append(i+1)
            sta_q.append(j+1)
            tradeAmt.append(df_side.loc[(df_side['mta_quantile'] == i+1) & (df_side['sta_quantile'] == j+1), 'order_amt'].sum()/n)
            
            df_local = df_side.loc[(df_side['mta_quantile'] == i+1) & (df_side['sta_quantile'] == j+1), ]
            df_local = df_local.loc[~df_local['1d_ret'].isna()]
            ret = (df_local['1d_ret'] * df_local['order_amt']).sum()/df_local['order_amt'].sum()
            ret_1d.append(ret)
            
            print(df_side)
            #sta_mean = df_local
            
            
            X = df_local[['alp_1d', 'prev_yHatBuy90' ]]
            Y = df_local['1d_ret']
            X = sm.add_constant(X) # adding a constant
     
            model = sm.OLS(Y, X).fit()
            r2 = model.rsquared
            mta_coef = model.params[1]
            sta_coef = model.params[2]
            
            R2.append(r2)
            MTA_coef.append(mta_coef)
            STA_coef.append(sta_coef)
            
            
            #tradeAmt_sta.append(df_side.loc[(df_side['mta_quantile'] == i+1) & (df_side['sta_quantile'] == j+1), 'order_amt'].sum())
            #print(i,j)
            #print(df_side.loc[(df_side['mta_quantile'] == i+1) & (df_side['sta_quantile'] == j+1), 'order_amt'].sum())
    df = pd.DataFrame({'mta_q':mta_q, 'sta_q':sta_q, 'tradeAmt': tradeAmt, 'STA_mean': STA_mean, 'ret_1d': ret_1d, 'r2': R2, 'MTA_coef': MTA_coef, 'STA_coef': STA_coef})
    return(df)

def summary_mta(df_side):
    tradeAmt = []
    ret_1d = []
    mta_q = []
    sta_q = []
    R2 = []
    MTA_coef = []
    STA_coef = []
    STA_mean = []
    
    if df_side['order_side'].iloc[0] == 1:
        df_side['sta'] = df_side['prev_yHatBuy90']
    elif df_side['order_side'].iloc[0] == 2:
        df_side['sta'] = df_side['prev_yHatSell90']
    
    for i in range(10):
        mta_q.append(i+1)
        tradeAmt.append(df_side.loc[(df_side['mta_quantile'] == i+1), 'order_amt'].sum()/n)
            
        df_local = df_side.loc[(df_side['mta_quantile'] == i+1), ]
        df_local = df_local.loc[~df_local['1d_ret'].isna()]
        ret = (df_local['1d_ret'] * df_local['order_amt']).sum()/df_local['order_amt'].sum()
        ret_1d.append(ret)
        print('!!!!!!!!!!!!!!!!!!!')
        print(df_side['order_side'].iloc[0])
        
        
        STA_mean.append(df_local['sta'].mean())   
            
        X = df_local[['alp_1d', 'prev_yHatBuy90' ]]
        Y = df_local['1d_ret']
        X = sm.add_constant(X) # adding a constant
     
        model = sm.OLS(Y, X).fit()
        r2 = model.rsquared
        mta_coef = model.params[1]
        sta_coef = model.params[2]
            
        R2.append(r2)
        MTA_coef.append(mta_coef)
        STA_coef.append(sta_coef)
        
        print(model.pvalues[1])
        print(model.pvalues[2])
            
            
            #tradeAmt_sta.append(df_side.loc[(df_side['mta_quantile'] == i+1) & (df_side['sta_quantile'] == j+1), 'order_amt'].sum())
            #print(i,j)
            #print(df_side.loc[(df_side['mta_quantile'] == i+1) & (df_side['sta_quantile'] == j+1), 'order_amt'].sum())
    df = pd.DataFrame({'mta_q':mta_q, 'tradeAmt': tradeAmt, 'STA_mean': STA_mean, 'ret_1d': ret_1d, 'r2': R2, 'MTA_coef': MTA_coef, 'STA_coef': STA_coef})
    return(df)
    

def summary_sta(df_side):
    tradeAmt = []
    ret_1d = []
    mta_q = []
    sta_q = []
    R2 = []
    MTA_coef = []
    STA_coef = []
    MTA_mean = []
    for i in range(10):
        sta_q.append(i+1)
        tradeAmt.append(df_side.loc[(df_side['sta_quantile'] == i+1), 'order_amt'].sum()/n)
            
        df_local = df_side.loc[(df_side['sta_quantile'] == i+1), ]
        df_local = df_local.loc[~df_local['1d_ret'].isna()]
        ret = (df_local['1d_ret'] * df_local['order_amt']).sum()/df_local['order_amt'].sum()
        ret_1d.append(ret)
            
        
        MTA_mean.append(df_local['alp_1d'].mean()) 
        
        
        X = df_local[['alp_1d', 'prev_yHatBuy90' ]]
        Y = df_local['1d_ret']
        X = sm.add_constant(X) # adding a constant
     
        model = sm.OLS(Y, X).fit()
        r2 = model.rsquared
        mta_coef = model.params[1]
        sta_coef = model.params[2]
            
        R2.append(r2)
        MTA_coef.append(mta_coef)
        STA_coef.append(sta_coef)
            
            
            #tradeAmt_sta.append(df_side.loc[(df_side['mta_quantile'] == i+1) & (df_side['sta_quantile'] == j+1), 'order_amt'].sum())
            #print(i,j)
            #print(df_side.loc[(df_side['mta_quantile'] == i+1) & (df_side['sta_quantile'] == j+1), 'order_amt'].sum())
    df = pd.DataFrame({'sta_q':sta_q, 'tradeAmt': tradeAmt, 'MTA_mean': MTA_mean, 'ret_1d': ret_1d, 'r2': R2, 'MTA_coef': MTA_coef, 'STA_coef': STA_coef})
    return(df)
    
    
before = datetime.datetime.now()          
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
    
    
    
    tuples = []
    for date in dates:
        tuples.append((date, order_side, order_cap))
    
    P = mp.Pool(96)
    df_list = P.map(stamta, tuples)
    P.close()
    P.join()
    
    df = pd.concat(df_list)
    print(df)
    
    #print(summary(df))
    summary_sta(df).to_csv('/data/home/jianw/OrderAnalysis/20211221_stamta_Interaction/sta_summary_buy.csv', index = False)


after = datetime.datetime.now()
print(after - before)