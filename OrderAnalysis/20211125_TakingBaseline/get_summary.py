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



def order_selection(df_side, method,demean, order_amt_all, order_side, top):
    '''
    if order_side == 1:
        df_side['sta'] = df_side['prev_yHatBuy90']
    elif order_side == 2:
        df_side['sta'] = df_side['prev_yHatSell90']
    '''
    
    
    #df_side['mta'] = np.where(order_side ==1, df_side['alp_1d'], -df_side['alp_1d'])
    
    if method == 'allDay':
        df_side = df_side.sort_values(by = ['alpha'], ascending = False)
        #df_side = df_side.sort_values(by = ['sta'], ascending = False)
        #df_side = df_side.sort_values(by = ['mta'], ascending = False)
        
        df_side['order_amt_all'] = df_side['order_amt'].sum()
        df_side['order_amt_cumsum'] = df_side['order_amt'].cumsum().shift()
        df_side['order_amt_cumsum'] = df_side['order_amt_cumsum'].fillna(0)
        
    elif method == 'minute':
        df_side = df_side.sort_values(by = ['interval','alpha'], ascending=[True, False])
        #df_side = df_side.sort_values(by = ['interval','sta'], ascending=[True, False])
        #df_side = df_side.sort_values(by = ['interval','mta'], ascending=[True, False])
        
        df_side['order_amt_all'] = df_side.groupby('interval')['order_amt'].transform('sum')
        df_side['order_amt_cumsum'] = df_side.groupby('interval')['order_amt'].transform(lambda x: x.cumsum().shift())
        df_side['order_amt_cumsum'] = df_side['order_amt_cumsum'].fillna(0)
        
    elif method == 'stock':
        df_side = df_side.sort_values(by = ['skey','alpha'], ascending=[True, False])
        #df_side = df_side.sort_values(by = ['skey','sta'], ascending=[True, False])
        #df_side = df_side.sort_values(by = ['skey','mta'], ascending=[True, False])
            
        df_side['order_amt_all'] = df_side.groupby('skey')['order_amt'].transform('sum')
        df_side['order_amt_cumsum'] = df_side.groupby('skey')['order_amt'].transform(lambda x: x.cumsum().shift())
        df_side['order_amt_cumsum'] = df_side['order_amt_cumsum'].fillna(0)
    
    if demean == 'minute':
        df_side['ret_1d_tradeAmt']  = (df_side['1d_ret'] * df_side['trade_amt'])
        df_side['ret_1d_tradeAmt_sum'] = df_side.groupby('interval')['ret_1d_tradeAmt'].transform('sum')
        df_side['tradeAmt_sum'] = df_side.groupby('interval')['trade_amt'].transform('sum')
        df_side['minute_mean'] = df_side['ret_1d_tradeAmt_sum'] / df_side['tradeAmt_sum']
        df_side['1d_ret_demean'] = df_side['1d_ret'] - df_side['minute_mean'] 
     
    df_side['threshold'] = df_side['order_amt_all'] * top
    #df_side.to_csv('sample.%s.%s.%s.%s.csv' %(date, order_side, method, demean), index = False)
    
    df_side_top = df_side.loc[df_side['order_amt_cumsum']<=df_side['threshold']]

        
    return(df_side_top)
    
def gen_summary(tuple):
    date, order_side, top, order_cap, method, demean  = tuple
    print(date)
    try:
        df = pd.read_parquet('/data/home/jianw/OrderAnalysis/20211125_TakingBaseline/concat_data/%s.parquet' %(date.replace('-', '')))
    except:
        print('%s data missing'%date)
        return None
    
    df = df.loc[~df['alpha'].isna(),].copy()
    '''
    df['safeBid1p'] = np.where(df.eval('bid1p==0'), df['ask1p'], df['bid1p'])
    df['safeAsk1p'] = np.where(df.eval('ask1p==0'), df['bid1p'], df['ask1p']) 
    df['adjMid'] = df.eval('(safeBid1p * ask1q + safeAsk1p * bid1q) / (bid1q + ask1q)')
    '''
    
    df['safeBid1p'] = np.where(df.eval('prev_bid1p==0'), df['prev_ask1p'], df['prev_bid1p'])
    df['safeAsk1p'] = np.where(df.eval('prev_ask1p==0'), df['prev_bid1p'], df['prev_ask1p']) 
    df['adjMid'] = df.eval('(safeBid1p * prev_ask1q + safeAsk1p * prev_bid1q) / (prev_bid1q + prev_ask1q)')
    
    
    
    df['90s_ret'] = np.nan
    df['90s_ret'] = np.where(df['order_side']==1, df['adjMidF90s']/df['adjMid'] - 1, df['90s_ret'])
    df['90s_ret'] = np.where(df['order_side']==2, 1 - df['adjMidF90s']/df['adjMid'], df['90s_ret'])
    #df['90s_ret'][(df['90s_ret'] == np.inf)|(df['90s_ret'] == -np.inf)] = 0
    
    #print(df)
    
    df['order_amt'] = np.minimum(df['order_amt'], order_cap)
    df['trade_amt'] = np.minimum(df['trade_qty_taking']*df['trade_price'], order_cap)
    # calculate minutely trade_amt
    df['minute_stock_trade_amt'] = df.groupby(['interval', 'skey'])['order_amt'].transform('sum')
    df_side = df.loc[df['order_side']==order_side,]
    order_amt_all = df_side['order_amt'].sum()
    trade_amt_all = df_side['trade_amt'].sum()
    fillrate_all = trade_amt_all/order_amt_all
    
    ## Select Top Orders 
    df_side_top = order_selection(df_side, method, demean, order_amt_all, order_side, top)
    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #print(df_side_top)
    
    df_minute_skey = df_side_top.groupby(['skey', 'interval']).agg({ 'minute_stock_trade_amt': 'mean', 'order_amt': 'sum', 'trade_amt': 'sum'}).reset_index()
    df_minute_skey['POV'] = df_minute_skey['order_amt'] / df_minute_skey['minute_stock_trade_amt'] 
    #print(df_minute_skey)
    df_minute_skey.to_csv('/data/home/jianw/OrderAnalysis/20211125_TakingBaseline/stats_summary/pov/pov.%s.%s.%s.csv' %(method, order_side,date), index = False)
    
    #df_side_top.to_csv('sell.csv', index = False)
    order_amt_top = df_side_top['order_amt'].sum()
    trade_amt_top = df_side_top['trade_amt'].sum()
    fillrate_top = trade_amt_top/order_amt_top
    
    ## calculate return
    #df_side_top['profit'] = df_side_top['1d_ret'] * df_side_top['trade_amt']
    if demean == 'minute':
        total_profit = (df_side_top['1d_ret_demean'] * df_side_top['trade_amt']).sum()
    else:
        total_profit = (df_side_top['1ret'] * df_side_top['trade_amt']).sum()
        
    total_sta = (df_side_top['90s_ret'] * df_side_top['trade_amt']).sum()
    return_bps = total_profit/trade_amt_top*10000
    return_sta_bps = total_sta/trade_amt_top*10000
    ## summary
    summary = pd.DataFrame([[date, order_side, order_amt_all, trade_amt_all, fillrate_all, order_amt_top, trade_amt_top,fillrate_top, total_profit,
                            return_bps, total_sta, return_sta_bps]],
                           columns=['date', 'order_side','order_amt_all', 'trade_amt_all','fillrate_all','order_amt_top', 'trade_amt_top',
                                    'fillrate_top', 'total_profit', 'return_bps', 'total_sta', 'return_sta_bps'])
    return(summary)


def saveSummary(sdate, edate, inSample, top, order_cap, order_side, method, ret):
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    #dates = date_list.to_list()
    if inSample == 1:
        IS = pd.read_csv("/data/home/jianw/L4PO/data/MTA_IS_Dates_CHN.csv")
        IS.columns = ['date']
        date_list = date_list.loc[date_list.isin(IS['date'])]
    dates = date_list.to_list()
    
    #print(dates)
    
    tuples = []
    for date in dates:
        tuples.append((date, order_side, top, order_cap, method, ret))
    print(tuples)
    
    P = mp.Pool(96)
    df_list = P.map(gen_summary, tuples)
    P.close()
    P.join()
    
    df = pd.concat(df_list)
    #df['order_side'] = order_side
    #df['date'] = dates
    df_summary = pd.DataFrame([['ALL', order_side, df['order_amt_all'].mean(), df['trade_amt_all'].mean(), df['trade_amt_all'].mean()/df['order_amt_all'].mean(),
      df['order_amt_top'].mean(), df['trade_amt_top'].mean(), df['trade_amt_top'].mean()/df['order_amt_top'].mean(), df['total_profit'].mean(), df['total_profit'].mean()/df['trade_amt_top'].mean()*10000, df['total_sta'].mean(), df['total_sta'].mean()/df['trade_amt_top'].mean()*10000, df['total_profit'].mean()/df['total_profit'].std()*(244**0.5)]],
                           columns=['date', 'order_side','order_amt_all', 'trade_amt_all','fillrate_all','order_amt_top', 'trade_amt_top',
                                    'fillrate_top', 'total_profit', 'return_bps', 'total_sta', 'return_sta_bps', 'SR'])
    df = df.append(df_summary)
    print(df)
    
    inSample_dic = {0: 'ALL',1:'IS', 2:'OOS'}
    df.to_csv('/data/home/jianw/OrderAnalysis/20211125_TakingBaseline/stats_summary/mta/%s.%s.%s.%s.%s.%s.csv'%(method,ret, order_side, sdate, edate, inSample_dic[inSample] ), index = False)


before = datetime.datetime.now()          
if __name__ == '__main__':
    
    top = 0.05
    order_cap = 1000000
    order_sides = [1,2]
    methods = ['allDay','minute', 'stock']
    rets = ['raw','minute']
    inSample = 1

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/sta/data_jian_newagg/'
    
    sdate = '2021-01-01'
    edate = '2021-06-30'
    
    saveSummary(sdate, edate, inSample, top, order_cap, order_sides[0], methods[1], rets[1])
    '''
    for order_side in order_sides:
        for method in methods:
            for ret in rets:
                saveSummary(sdate, edate, inSample, top, order_cap, order_side, method, ret)
    '''
    
    '''
    

    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    #dates = date_list.to_list()
    if inSample == 1:
        IS = pd.read_csv("/data/home/jianw/L4PO/data/MTA_IS_Dates_CHN.csv")
        IS.columns = ['date']
        date_list = date_list.loc[date_list.isin(IS['date'])]
    dates = date_list.to_list()
    print(dates)
    
    
    
    tuples = []
    for date in dates:
        tuples.append((date, order_side, top, order_cap, method))
    #print(tuples)
    
    P = mp.Pool(96)
    df_list = P.map(gen_summary, tuples)
    P.close()
    P.join()
    
    df = pd.concat(df_list)
    df_summary = pd.DataFrame([['ALL', order_side, df['order_amt_all'].mean(), df['trade_amt_all'].mean(), df['trade_amt_all'].mean()/df['order_amt_all'].mean(),
      df['order_amt_top'].mean(), df['trade_amt_top'].mean(), df['trade_amt_top'].mean()/df['order_amt_top'].mean(), df['total_profit'].mean(), df['total_profit'].mean()/df['trade_amt_top'].mean()*10000, df['total_profit'].mean()/df['total_profit'].std()*(244**0.5)]],
                           columns=['date', 'order_side','order_amt_all', 'trade_amt_all','fillrate_all','order_amt_top', 'trade_amt_top',
                                    'fillrate_top', 'total_profit', 'return_bps', 'SR'])
    df = df.append(df_summary)
    print(df)
    
    df.to_csv('/data/home/jianw/OrderAnalysis/20211125_TakingBaseline/stats_summary/%s.%s.%s.%s.csv'%(method, order_side, sdate, edate), index = False)
    '''

    
    

after = datetime.datetime.now()
print(after - before)