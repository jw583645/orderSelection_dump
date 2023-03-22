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



def summary_mta(decile):
    i = decile
    tradeAmt = []
    ret_30m = []
    mta_q = []
    sta_q = []
    R2 = []
    MTA_coef = []
    STA_coef = []
    Spread_coef = []
    STA_mean = []
    MTA_mean = []
    Spread_mean = []
    
    '''
    if (df_side['order_side'].mean() > 1) & (df_side['order_side'].mean() < 2):
        df_side['sta'] = df_side['prev_yHatMid90']    
    elif df_side['order_side'].iloc[0] == 1:
        df_side['sta'] = df_side['prev_yHatBuy90']
    elif df_side['order_side'].iloc[0] == 2:
        df_side['sta'] = df_side['prev_yHatSell90']
    '''
    mta_q.append(i)
    
    df_local = pd.read_parquet('/data/home/jianw/OrderAnalysis/20220117_30min/decile_data/orderside%s/decile%s.parquet'%(order_side, i))
    df_local = df_local.loc[~((df_local['time']> 142700000000))]
    #df_local = df_local.loc[df_local['mid_price']<20]
    df_local = df_local.loc[~df_local['30m_ret'].isna()]
    df_local['sta'] = df_local['prev_yHatMid90']  
    
    
    
    ## use trade price to calculate 1d ret
    '''
    df_local['1d_ret'] = np.nan  
    df_local['1d_ret'] = np.where(df_local['order_side'] == 1, (np.log(df_local['stk_dv_ND1_adj']/df_local['trade_price']) - df_local['beta'] * np.log(df_local['idx_dv_ND1']/df_local['ic'])), df_local['1d_ret'])
    df_local['1d_ret'] = np.where(df_local['order_side'] == 2, (np.log(df_local['trade_price']/df_local['stk_dv_ND1_adj']) - df_local['beta'] * np.log(df_local['ic']/df_local['idx_dv_ND1'])), df_local['1d_ret'])
    '''
    
    
    tradeAmt.append(df_local['order_amt'].sum()/n)

        
    ret = (df_local['30m_ret'] * df_local['order_amt']).sum()/df_local['order_amt'].sum()
    ret_30m.append(ret)
                
    STA_mean.append(df_local['sta'].mean())  
    MTA_mean.append(df_local['alp_30m'].mean())  
    Spread_mean.append(df_local['halfSpread'].mean())  
            
    X = df_local[['alp_30m', 'sta', 'halfSpread']]
    #X = df_local[['alp_1d', 'sta']]
    Y = df_local['30m_ret']
    X = sm.add_constant(X) # adding a constant
     
    model = sm.OLS(Y, X).fit()
    #model = sm.WLS(Y, X, weights = df_local['trade_amt'].values ).fit()
    print(model.summary())
    r2 = model.rsquared
    mta_coef = model.params[1]
    sta_coef = model.params[2]
    spread_coef = model.params[3]
            
    R2.append(r2)
    MTA_coef.append(mta_coef)
    STA_coef.append(sta_coef)
    Spread_coef.append(spread_coef)
        
    #print(model.pvalues[1])
    #print(model.pvalues[2])
            
            
            #tradeAmt_sta.append(df_side.loc[(df_side['mta_quantile'] == i+1) & (df_side['sta_quantile'] == j+1), 'order_amt'].sum())
            #print(i,j)
            #print(df_side.loc[(df_side['mta_quantile'] == i+1) & (df_side['sta_quantile'] == j+1), 'order_amt'].sum())
    df = pd.DataFrame({'mta_q':mta_q, 'tradeAmt': tradeAmt, 'MTA_mean': MTA_mean, 'STA_mean': STA_mean, 'Spread_mean': Spread_mean, 'ret_30m': ret_30m, 'r2': R2, 'MTA_coef': MTA_coef, 'STA_coef': STA_coef, 'Spread_coef': Spread_coef})
    #df = pd.DataFrame({'mta_q':mta_q, 'tradeAmt': tradeAmt, 'MTA_mean': MTA_mean, 'STA_mean': STA_mean, 'ret_1d': ret_1d, 'r2': R2, 'MTA_coef': MTA_coef, 'STA_coef': STA_coef})
    #print(i)
    #print(df_local['1d_ret'].describe())
    #print(df_local['halfSpread'].describe())
    #print(df_local[['1d_ret', 'alp_1d', 'sta', 'halfSpread']].cov())
    
    #X = df_local[['alp_1d', 'sta', 'halfSpread', 'mid_price']]
    #X = df_local[['alp_1d', 'sta']]
    #Y = df_local['1d_ret']
    #X = sm.add_constant(X) # adding a constant
     
    #model = sm.OLS(Y, X).fit()
    #model = sm.WLS(Y, X, weights = df_local['order_amt'] ).fit()
    #print(model.summary())
    
    return(df)
    
    

    
    
before = datetime.datetime.now()          
if __name__ == '__main__':
    
    order_cap = 1000000
    order_side = 1

    inSample = 1

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/sta/data_jian_newagg/'
    
    sdate = '2021-01-01'
    edate = '2021-10-29'
    
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
    

    
    P = mp.Pool(96)
    df_list = P.map(summary_mta, [1,2,3,4,5,6,7,8,9,10])
    P.close()
    P.join()
    
    df = pd.concat(df_list)
    print(df)
    
    #print(summary(df))
    
    
    df.to_csv('/data/home/jianw/OrderAnalysis/20220117_30min/decile_data//drop30_ols_constant_spread_mta_summary_orderside%s.%s.%s.csv' %(order_side, sdate, edate), index = False)


after = datetime.datetime.now()
print(after - before)