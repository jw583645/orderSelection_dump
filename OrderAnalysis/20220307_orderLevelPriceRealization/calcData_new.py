import os, sys
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from datetime import date
from datetime import datetime
sys.path.insert(0, '/data/home/jianw/dfs_sh_client/')
import jdfs
from AShareReader import AShareReader



def loadAlphaF90(date):
    # alpha data 
    alphapool = 'sta_alpha_eq_cn' 
    alphans = 'sta_90_comb_1_1_mbd' 
    alphaPath = f"/sta_alpha_eq_cn/sta_90_comb_1_1_mbd/{idx}/sta{date}.parquet"
    alpha = jdfs.read_file(alphaPath, alphapool, alphans) 
    alpha.rename(columns={'yHatBuy':'yHatBuy90', 'yHatSell':'yHatSell90'}, inplace=True)
    
    
    return alpha


def concat(date):
    print(date)
    ## load summary data
    df_l = []
    file_path = "/data/home/jianw/OrderAnalysis/20220307_orderLevelPriceRealization/data/rawData/%s/%s/" %(idx, date.replace('-', ''))
    for file in os.listdir(file_path):
        #print(file)
        try:
            df = pd.read_parquet(file_path + file)
            df_l.append(df)
        except:
            pass
    df = pd.concat(df_l)
    df = df.loc[(df['time']< 145700000000) & (df['time']> 93000000000) ]
    #df = df.loc[(df['interval']> 144500000000) & (df['interval']<= 145600000000)]
    df['beta'] = df['beta'].fillna(1)
    sta = loadAlphaF90(date.replace('-', ''))
    sta['ordering_orange'] = sta['ordering']
    df = df.merge(sta[['skey', 'ordering_orange', 'yHatBuy90']], on = ['skey', 'ordering_orange'],how = 'left')
    
    print(df)
    print(df.columns)
    
    try:
        os.makedirs("/data/home/jianw/OrderAnalysis/20220307_orderLevelPriceRealization/data/daily/%s" %(idx))
    except:
        pass
    df.to_parquet("/data/home/jianw/OrderAnalysis/20220307_orderLevelPriceRealization/data/daily/%s/%s.parquet" %(idx, date))
    
    '''
    df['topBar'] = df.groupby('interval')['alp_1d'].transform(lambda x: np.quantile(x, 1-top))
    df_top = df.loc[df['alp_1d']>=df['topBar']]
    df_top['retTop'] = df_top.groupby('interval')['ret_30s'].transform(lambda x: np.quantile(x, 0.8))
    df_top['retBottom'] = df_top.groupby('interval')['ret_30s'].transform(lambda x: np.quantile(x, 0.2))
    df_top = df_top.loc[(df_top['ret_30s'] >= df_top['retBottom']) & (df_top['ret_30s'] <= df_top['retTop'])]
    #df_top.to_csv('/data/home/jianw/OrderAnalysis/20220222_modelFitting/mtaRealization/topName/%s/top%s_%s.csv' %(idx, top, date), index = False)
    
    q50_30s = np.quantile(df_top['ret_30s'], 0.50) * 10000
    q50_1d = np.quantile(df_top['ret_1d'], 0.50) * 10000
    q50_30s_1d = np.quantile(df_top['ret_30s_1d'], 0.50) * 10000
    

    mean_1d = df_top['ret_1d'].mean() * 10000
    mean_30s = df_top['ret_30s'].mean() * 10000
    mean_30s_1d = df_top['ret_30s_1d'].mean() * 10000
    mean_50s = df_top['ret_50s'].mean() * 10000
    mean_50s_1d = df_top['ret_50s_1d'].mean() * 10000
    
    
    #print(np.quantile(df_top['ret_1d'], 0.50) * 10000 , q50_top20_1d, df_top['ret_1d'].mean()*10000)
    
    #df = pd.DataFrame({'date': date, 'median_30s': q50_30s, 'mean_30s': mean_30s, 'median_1d': q50_1d, 'mean_1d': mean_1d, 'median_30s_1d': q50_30s_1d, 'mean_30s_1d': mean_30s_1d}, index = [0])
    df = pd.DataFrame({'date': date, 'mean_30s': mean_30s, 'mean_50s': mean_50s,  'mean_1d': mean_1d, 'mean_30s_1d': mean_30s_1d, 'mean_50s_1d': mean_50s_1d}, index = [0])
    return(df)
    '''
def getSummary(date):
    print(date)
    df = pd.read_parquet("/data/home/jianw/OrderAnalysis/20220307_orderLevelPriceRealization/data/daily/%s/%s.parquet" %(idx, date))
    df = df.loc[(df['orange_minute_2half'] == 1) & (~df['alp_1d'].isna())]
    
    #print(df)
    df_skey_interval = df.groupby(['interval', 'skey'])['alp_1d'].agg('mean').reset_index()
    df_interval = df_skey_interval.groupby(['interval'])['alp_1d'].agg(lambda x: np.quantile(x, 1-top)).reset_index()
    df_interval.rename(columns = {'alp_1d': 'topBar'}, inplace = True)
    df = df.merge(df_interval, on = 'interval', how = 'left')

    ## time of the day
    df = df.loc[(df['time'] > 100000000000)]
    
    df['topBar'] = df.groupby('interval')['alp_1d'].transform(lambda x: np.quantile(x, 1-top))
    df_top = df.loc[df['alp_1d']>=df['topBar']]
    
    ## buy taking/sell taking
    if order_type == 'buyTaking':
        df_top = df_top.loc[df_top['buyTaking']]
    elif order_type == 'sellTaking':
        df_top = df_top.loc[~df_top['buyTaking']]
    
    
    
    df_top['trade_amt'] = df_top['trade_price'] * df_top['trade_qty']
    df_top['profit'] = df_top['trade_amt'] * df_top['ret_trade_1d']
    
    dfSummary = pd.DataFrame({'date': date, 'trade_amt': df_top['trade_amt'].sum(), 'ret': df_top['profit'].sum()/df_top['trade_amt'].sum()}, index = [0])
    realizedBars = [-10000, -100, -50, -20, 0,20, 50, 100, 10000]
    for i in range(len(realizedBars)-1):
        #print(realizedBars[i], realizedBars[i+1])
        #df_local = df_top.loc[(df_top['ret_orange_trade']>realizedBars[i]/10000) & (df_top['ret_orange_trade']<=realizedBars[i+1]/10000)] 
        #df_local = df_top.loc[(df_top['ret_ask_trade']>realizedBars[i]/10000) & (df_top['ret_ask_trade']<=realizedBars[i+1]/10000)] 
        if order_type == 'buyTaking':
            df_local = df_top.loc[(df_top['ret_ask_trade']>realizedBars[i]/10000) & (df_top['ret_ask_trade']<=realizedBars[i+1]/10000)] 
        elif order_type == 'sellTaking':
            df_local = df_top.loc[(df_top['ret_bid_trade']>realizedBars[i]/10000) & (df_top['ret_bid_trade']<=realizedBars[i+1]/10000)] 
        dfSummary['trade_amt_%s' %i] = df_local['trade_amt'].sum() / df_top['trade_amt'].sum()
        dfSummary['ret_%s' %i] = df_local['profit'].sum() / df_local['trade_amt'].sum()
        dfSummary['alp_1d_%s' %i] = df_local.groupby(['skey', 'interval'])['alp_1d'].agg('mean').reset_index()['alp_1d'].mean()
        dfSummary['sta_%s' %i] = (df_local['yHatBuy90'] * df_local['trade_amt']).sum() / df_local['trade_amt'].sum() 
    return(dfSummary)
    
    
    

before = datetime.now()     
if __name__ == '__main__':
    RCH_DIR = '/data/rch/'
    #data_path = '/home/jianw/monetization_research/rawData_jian/data_jian_20220119/if/'
    sdate = '2021-01-01'
    edate = '2021-11-29'
    
    Idxs = ['IF', 'IC', 'CSI1000', 'CSIRest']
    #Idxs = ['IC']
    #idx = 'IF'
    top = 0.2
    Order_types = ['buyTaking', 'sellTaking']
    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    '''
    #getSummary('2021-01-04')
    P = mp.Pool(96)
    #df_l = P.map(getSummary, dates)
    P.map(concat, dates)
    P.close()
    P.join()
    '''
    
    for idx in Idxs:
        for order_type in Order_types:
            P = mp.Pool(96)
            #P.map(loadAlphaF90, list(set(daily_data['date'])))
            df_l = P.map(getSummary, dates)
            #P.map(concat, dates)
            P.close()
            P.join()
            
            df = pd.concat(df_l)
            df_summary = pd.DataFrame({'date': 'ALL', 'trade_amt': df['trade_amt'].mean(), 'ret': (df['trade_amt'] * df['ret']).sum() / df['trade_amt'].sum()}, index = [0])
            for i in range(8):
                df_summary['trade_amt_%s'%i] = (df['trade_amt_%s'%i] * df['trade_amt']).sum() / df['trade_amt'].sum()
                df_summary['ret_%s'%i] = (df['trade_amt_%s'%i] * df['trade_amt'] * df['ret_%s'%i]).sum() / (df['trade_amt_%s'%i] * df['trade_amt']).sum()
                df_summary['alp_1d_%s'%i] = df['alp_1d_%s'%i].mean()
            df = df.append(df_summary)
            df.to_csv('/data/home/jianw/OrderAnalysis/20220307_orderLevelPriceRealization/data/summaryData_20220318/%s.after30_summary.%s.%s.%s.csv' %(idx, order_type, sdate, edate), index =                 False)
        
    
    
    
    
    '''
    for idx in Idxs:
        P = mp.Pool(96)
        #P.map(loadAlphaF90, list(set(daily_data['date'])))
        df_l = P.map(getSummary, dates)
        P.close()
        P.join()
        
        df = pd.concat(df_l)
        df = df.append(df.mean(numeric_only=True), ignore_index=True)
        print(df)
        df.to_csv('/data/home/jianw/OrderAnalysis/20220303_mtaRealization/data/summaryData/%s.top%s.bottom50sRet.csv' %(idx, top), index = False)
    '''
    
    
    

    
    
after = datetime.now()
print(after - before)