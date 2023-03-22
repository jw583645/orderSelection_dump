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
        #df = df.loc[(df['ask1p']!=0) & (df['bid1p']!=0)]
        df = df.loc[(df['prev_ask1p']!=0) & (df['prev_bid1p']!=0)]
    except:
        print('%s data missing'%date)
        return None
        
    df = df.loc[~df['alpha'].isna(),].copy()
    df['order_amt'] = np.minimum(df['order_amt'], order_cap)
    df['trade_amt'] = np.minimum(df['trade_qty_taking']*df['trade_price'], order_cap)
    
    
    df['safeBid1p'] = np.where(df.eval('bid1p==0'), df['ask1p'], df['bid1p'])
    df['safeAsk1p'] = np.where(df.eval('ask1p==0'), df['bid1p'], df['ask1p'])
    df['mid_price'] = (df['safeBid1p'] + df['safeAsk1p']) / 2
    df['halfSpread'] = 0.5 * (df['safeAsk1p'] - df['safeBid1p']) / df['mid_price']
    
    # calculate mid-to-mid sta
    df['stahat_mid_90s'] = (df['prev_bid1p'] / ( 1 + df['prev_yHatSell90']) + df['prev_ask1p'] * ( 1 + df['prev_yHatBuy90'])) / 2
    #df['mid_90s'] = (df['prev_bid1p'] + df['prev_ask1p'])/2
    df['prev_yHatMid90'] = (df['stahat_mid_90s']/df['mid_price']) - 1
    
    
    

        
        
    df['1d_ret'] = np.nan  
    df['1d_ret'] = np.where(df['order_side'] == 1, (np.log(df['stk_dv_ND1_adj']/df['trade_price']) - df['beta'] * np.log(df['idx_dv_ND1']/df['ic'])), df['1d_ret'])
    df['1d_ret'] = np.where(df['order_side'] == 2, (np.log(df['trade_price']/df['stk_dv_ND1_adj']) - df['beta'] * np.log(df['ic']/df['idx_dv_ND1'])), df['1d_ret'])
        
    #df['1d_ret'] = (np.log(df['stk_dv_ND1_adj']/df['mid_price']) - df['beta'] * np.log(df['idx_dv_ND1']/df['ic']))
    df['1d_ret'] = np.where(df['trade_price'].isna(), np.nan, df['1d_ret']  )
    #df['1d_ret'] = np.where(df['order_side'] == 1, (np.log(df['stk_dv_ND1_adj']/df['mid_price']) - df['beta'] * np.log(df['idx_dv_ND1']/df['ic'])), df['1d_ret'])
    #df['1d_ret'] = np.where(df['order_side'] == 2, (np.log(df['mid_price']/df['stk_dv_ND1_adj']) - df['beta'] * np.log(df['ic']/df['idx_dv_ND1'])), df['1d_ret'])
    df_side = df.loc[df['order_side'] == order_side]
    #df_side['1d_ret'] = np.where(df_side['order_side'] == 1, df_side['1d_ret'], -df_side['1d_ret'] )
    
    df_side['order_amt_all'] = df_side.groupby('interval')['order_amt'].transform('sum')
    
    df_side = df_side.sort_values(by = ['interval','alp_1d', 'clockAtArrival'], ascending=[True, True, True]) 
    '''
    if order_side == 1:
        df_side = df_side.sort_values(by = ['interval','alp_1d', 'clockAtArrival'], ascending=[True, True, True]) 
    elif order_side == 2:
        df_side = df_side.sort_values(by = ['interval','alp_1d', 'clockAtArrival'], ascending=[True, True, True]) 
    elif order_side == 0:
        df_side = df_side.sort_values(by = ['interval','alp_1d', 'clockAtArrival'], ascending=[True, True, True]) 
    '''  
    df_side['order_amt_cumsum_mta'] = df_side.groupby('interval')['order_amt'].transform('cumsum')
    df_side['mta_quantile'] = (np.ceil(df_side['order_amt_cumsum_mta'] / df_side['order_amt_all'] * 10)).astype(int)
    
    
    df_side = df_side.sort_values(by = ['interval','prev_yHatMid90', 'clockAtArrival'], ascending=[True, True, True])  
    '''
    if order_side == 1:
        df_side = df_side.sort_values(by = ['interval','prev_yHatBuy90', 'clockAtArrival'], ascending=[True, True, True])    
    elif order_side == 2:
        df_side = df_side.sort_values(by = ['interval','prev_yHatSell90', 'clockAtArrival'], ascending=[True, True, True])
    elif order_side == 0:
        df_side = df_side.sort_values(by = ['interval','prev_yHatMid90', 'clockAtArrival'], ascending=[True, True, True])  
    '''
    df_side['order_amt_cumsum_sta'] = df_side.groupby('interval')['order_amt'].transform('cumsum')
    df_side['sta_quantile'] = (np.ceil(df_side['order_amt_cumsum_sta'] / df_side['order_amt_all'] * 10)).astype(int)
    
    #print('!!!!!!!!!!!!!!!!!')
    #print(df_side)
    '''
    if date == '2021-04-23':
        df_side.to_parquet('df_orderside0.%s.parquet' %date)
    '''
    return(df_side)
    


def summary_mta(df_side):
    tradeAmt = []
    ret_1d = []
    mta_q = []
    sta_q = []
    R2 = []
    MTA_coef = []
    STA_coef = []
    Spread_coef = []
    STA_mean = []
    MTA_mean = []
    
    df_side['sta'] = df_side['prev_yHatMid90']  
    
    for i in range(10):
        mta_q.append(i+1)
        tradeAmt.append(df_side.loc[(df_side['mta_quantile'] == i+1), 'order_amt'].sum()/n)
            
        df_local = df_side.loc[(df_side['mta_quantile'] == i+1), ]
        df_local = df_local.loc[~df_local['1d_ret'].isna()]
        
        df_local.to_parquet('/data/home/jianw/OrderAnalysis/20211221_stamta_Interaction/decile_data/orderside%s/decile%s.parquet'%(order_side, i+1))
        
        ret = (df_local['1d_ret'] * df_local['order_amt']).sum()/df_local['order_amt'].sum()
        ret_1d.append(ret)
        #print('!!!!!!!!!!!!!!!!!!!')
        #print(df_side['order_side'].iloc[0])
        
        
        STA_mean.append(df_local['sta'].mean())  
        MTA_mean.append(df_local['alp_1d'].mean())  
            
        #X = df_local[['alp_1d', 'sta', 'halfSpread']]
        X = df_local[['alp_1d', 'sta']]
        Y = df_local['1d_ret']
        X = sm.add_constant(X) # adding a constant
     
        model = sm.OLS(Y, X).fit()
        #model = sm.WLS(Y, X, weights = df_local['order_amt'] ).fit()
        r2 = model.rsquared
        mta_coef = model.params[1]
        sta_coef = model.params[2]
        #spread_coef = model.params[3]
            
        R2.append(r2)
        MTA_coef.append(mta_coef)
        STA_coef.append(sta_coef)

    #df = pd.DataFrame({'mta_q':mta_q, 'tradeAmt': tradeAmt, 'MTA_mean': MTA_mean, 'STA_mean': STA_mean, 'ret_1d': ret_1d, 'r2': R2, 'MTA_coef': MTA_coef, 'STA_coef': STA_coef, 'Spread_coef': Spread_coef})
    df = pd.DataFrame({'mta_q':mta_q, 'tradeAmt': tradeAmt, 'MTA_mean': MTA_mean, 'STA_mean': STA_mean, 'ret_1d': ret_1d, 'r2': R2, 'MTA_coef': MTA_coef, 'STA_coef': STA_coef})
    return(df)
    

    
    
before = datetime.datetime.now()          
if __name__ == '__main__':
    
    order_cap = 1000000
    order_side = 2

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
    summary_mta(df).to_csv('/data/home/jianw/OrderAnalysis/20211221_stamta_Interaction/constant_mta_summary_orderside%s.%s.%s.csv' %(order_side, sdate, edate), index = False)


after = datetime.datetime.now()
print(after - before)