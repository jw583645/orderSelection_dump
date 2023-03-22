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

def date_transfer(date, direction):
    if direction == 1:
        date = str(date)
        return(date[0:4] + '-' + date[4:6] + '-' + date[6:8])

def orange_interval(time):
    import datetime
    x = (time/1e6).astype(int).astype(str)
    y = np.where(x.str.len() < 6 , '0' + x.str.slice(0,1) + ':' + x.str.slice(1,3) + ':' + x.str.slice(3,5),
        x.str.slice(0,2) + ':' + x.str.slice(2,4) + ':' + x.str.slice(4,6))  
    z = pd.Series(((pd.to_datetime(y) - datetime.timedelta(seconds=20)).time).astype(str))
    interval = (z.str.replace(':','').astype(int)/100).astype(int)*1e8
    
    interval = interval.replace({125900000000: 113000000000, 130000000000: 113000000000})
    return(interval)

def loadAlphaF90(date, skey):
    # alpha data 
    alphapool = 'sta_alpha_eq_cn' 
    alphans = 'sta_90_comb_1_1_mbd' 
    alphaPath = f"/sta_alpha_eq_cn/sta_90_comb_1_1_mbd/CSI1000/sta{date}.parquet"
    alpha = jdfs.read_file(alphaPath, alphapool, alphans) 
    alpha.rename(columns={'yHatBuy':'yHatBuy90', 'yHatSell':'yHatSell90'}, inplace=True)
    
    
    return alpha

def loadTrade(date, skey):
    tradepool = 'com_md_eq_cn'
    tradens = 'md_trade'
    tradePath = f"/com_md_eq_cn/md_trade/{date}/{skey}.parquet"
    trade = jdfs.read_file(tradePath, tradepool,tradens)
    trade.to_csv('/data/home/jianw/OrderAnalysis/20220222_modelFitting/mbd.csv')
    trade = trade.loc[trade['trade_type'] == 1]
    trade['ApplSeqNum'] = np.maximum(trade['BidApplSeqNum'], trade['OfferApplSeqNum'])
    return trade[['ApplSeqNum', 'trade_price', 'trade_qty']]
    
    
def loadIndex(date):
    index_skey = {'IF': 1000300, 'IC': 1000905, 'CSI1000': 1000852, 'CSIRest': 1000852}
    reader = AShareReader()
    index = reader.Read_Stock_Tick('com_md_eq_cn', 'md_index', date, date, stock_list=[index_skey[idx]])
    index.drop_duplicates(['skey','date','time'], keep='last', inplace=True)
    index['second'] = (index['time'] / 1000000).astype('int64')
    index.rename(columns={'close':'idx'}, inplace=True) 
    return(index[['date','second', 'idx']])

def loadMbd(pair):
    # mbd data 
    date, skey = pair
    mbdpool = 'com_md_eq_cn' 
    mbdns = 'md_snapshot_mbd' 
    mbdPath = f"/com_md_eq_cn/md_snapshot_mbd/{date}/{skey}.parquet" 
    df = jdfs.read_file(mbdPath, mbdpool, mbdns)
    if df is None:
        print('missing data: %s, %s' %(date, skey))
        return None
    trade = loadTrade(date, skey)
    df = df.merge(trade, on = 'ApplSeqNum', how = 'left')

    df['safeBid1p'] = np.where(df.eval('bid1p==0'), df['ask1p'], df['bid1p'])
    df['safeAsk1p'] = np.where(df.eval('ask1p==0'), df['bid1p'], df['ask1p'])
    df['adjMid'] = df.eval('(safeBid1p * ask1q + safeAsk1p * bid1q) / (bid1q + ask1q)')
    df['mid'] = (df['safeBid1p'] + df['safeAsk1p']) / 2
    
    df['interval'] = orange_interval(df['time'])
    df['second'] = (df['time'] / 1000000).astype('int64')
    index = loadIndex(date)
    df = pd.merge(df, index, how='left', on=['date', 'second'], validate='many_to_one')
    df = df[['skey', 'date', 'ordering',  'time', 'interval', 'mid', 'adjMid', 'trade_price', 'trade_qty','cum_amount', 'idx']]
    df['trade_price'].fillna(method="ffill", inplace = True)
    df['trade_price'].fillna(method="bfill", inplace = True)
    df['trade_qty'].fillna(0, inplace = True)
    df['trade_amt'] = df['trade_qty'] * df['trade_price']
    #df.to_csv('/data/home/jianw/OrderAnalysis/20220222_modelFitting/moutai.csv', index = False)
    
    n = len(df)
    df['minute_30s'] = df['interval'] + 30000000
    df['minute_30s'] = df['minute_30s'].replace({113030000000: 130030000000})
    df['minute_50s'] = df['interval'] + 50000000
    df['minute_50s'] = df['minute_50s'].replace({113050000000: 130050000000})
    
    df['orange_minute_2half'] = (df['time'] > df['minute_50s'] ).astype(int)
    
    
    #df.to_csv('/data/home/jianw/OrderAnalysis/20220222_modelFitting/moutai.csv', index = False)
    
    
    idx_orange = np.minimum( np.searchsorted(df['time'], df['interval'] ), n-1)
    df['idx_orange'] = df.iloc[idx_orange]['idx'].reset_index(drop = True)
    df['mid_orange'] = df.iloc[idx_orange]['mid'].reset_index(drop = True)
    idx_30s = np.minimum( np.searchsorted(df['time'], df['minute_30s'] ), n-1)
    df['idx_30s'] = df.iloc[idx_30s]['idx'].reset_index(drop = True)
    df['mid_30s'] = df.iloc[idx_30s]['mid'].reset_index(drop = True)
    idx_50s = np.minimum( np.searchsorted(df['time'], df['minute_50s'] ), n-1)
    df['idx_50s'] = df.iloc[idx_50s]['idx'].reset_index(drop = True)
    df['mid_50s'] = df.iloc[idx_50s]['mid'].reset_index(drop = True)
    
    ## include mta
    mta = read_mta(date_transfer(date, 1))
    df = df.merge(mta, on = ['interval', 'skey'], how = 'left')
    df['ret_trade_1d'] = np.log(df['stk_dv_ND1_adj']/df['trade_price']) - df['beta'] * np.log(df['idx_dv_ND1']/df['idx'])
    df['trade_amt2'] = df['trade_amt'] * df['orange_minute_2half']
    df['profit2'] = df['ret_trade_1d'] * df['trade_amt2']
    df['profit'] = df['ret_trade_1d'] * df['trade_amt']
    
    
    df.to_csv('/data/home/jianw/OrderAnalysis/20220222_modelFitting/swy.csv', index = False)
    #print(df)
    ## group by orange minute
    df = df.groupby(['skey','date',  'interval']).agg({'cum_amount':'max', 'alp_1d': 'mean', 'mid_orange' : 'mean', 'idx_orange': 'mean', 'mid_30s': 'mean','idx_30s': 'mean', 'mid_50s': 'mean', 'idx_50s': 'mean', 
    'beta': 'mean', 'idx_dv_ND1': 'mean', 'stk_dv_ND1_adj': 'mean', 'profit': 'sum', 'trade_amt': 'sum', 'profit2': 'sum', 'trade_amt2': 'sum'}).reset_index()
    df['ret_trade_1d'] = df['profit']/df['trade_amt']
    df['ret_trade_1d2'] = df['profit2']/df['trade_amt2']
    df['trade_amt'] = df['cum_amount'].diff()
    df = df.loc[~df['alp_1d'].isna()]
    df['ret_1d'] = np.log(df['stk_dv_ND1_adj']/df['mid_orange']) - df['beta'] * np.log(df['idx_dv_ND1']/df['idx_orange'])
    df['ret_30s'] = np.log(df['mid_30s']/df['mid_orange']) - df['beta'] * np.log(df['idx_30s']/df['idx_orange'])
    df['ret_50s'] = np.log(df['mid_50s']/df['mid_orange']) - df['beta'] * np.log(df['idx_50s']/df['idx_orange'])
    df['ret_30s_1d'] = np.log(df['stk_dv_ND1_adj']/df['mid_30s']) - df['beta'] * np.log(df['idx_dv_ND1']/df['idx_30s'])
    df['ret_50s_1d'] = np.log(df['stk_dv_ND1_adj']/df['mid_50s']) - df['beta'] * np.log(df['idx_dv_ND1']/df['idx_50s'])
    #df.to_csv('/data/home/jianw/OrderAnalysis/20220222_modelFitting/moutai.csv', index = False)
    ## save files
    
    try:
        os.makedirs('/data/home/jianw/OrderAnalysis/20220222_modelFitting/mtaRealization/%s/%s' %(idx, date))
    except:
        pass
    df.to_parquet('/data/home/jianw/OrderAnalysis/20220222_modelFitting/mtaRealization/%s/%s/%s.parquet' %(idx, date, skey))
    
    #print(df)
    #return(df)

def read_mta(date):
    mta =  pd.read_parquet('/data/home/jianw/monetization_research/adjDivPrice/mret_fwd.CHNUniv.EqALL4.1MIN.%s.parquet' %(date))
    idx_stock = {'IF': 'SZ000001', 'IC': 'SZ000009', 'CSI1000': 'SZ000011', 'CSIRest': 'SZ000011'}
    ic_dclose_ND1 = mta.loc[(mta['skey'] == idx_stock[idx]) & (mta['interval'] == '09:31:00'), 'idx_dclose_ND1'].iloc[0]
    #print(ic_dclose_ND1)
    mta = mta.loc[(mta['idx_dclose_ND1'] <= ic_dclose_ND1 + 1e-6)&(mta['idx_dclose_ND1'] >= ic_dclose_ND1 - 1e-6),]
    # read 30m mta
    #mta_30m = pd.read_csv('/data/home/jianw/mta_eq_cn/data/rch/eq/sig/CHNUniv.EqALL4/1MIN/CHNOR001/1.3.1c/alp.hzn30m/alp.CHNOR001.%s.csv.gz' %date)
    #mta_30m = mta_30m.rename(columns = {'yhat.mret_fT0_30m': 'alp_30m'})
    #mta = mta.merge(mta_30m, on = ['skey', 'date' ,'interval'], how = 'left')

    #print(date, len(list(set(mta['skey']))))
    mta['interval'] = (mta['interval'].str.replace(':', '')).astype(int)*1e6
    mta['skey'] = np.where(mta['skey'].str.slice(0,2) == 'SH', '1' + mta['skey'].str.slice(2), '2' + mta['skey'].str.slice(2)).astype(int)
    #return mta[['skey', 'interval', 'alp_1d', 'alp_2d', 'alp_30m', 'beta', 'idx_dv_ND1', 'stk_dv_ND1', 'idx_N30min', 'stk_N30min', 'stk_dclose_ND1', 'idx_dv_ND2', 'stk_dv_ND2', 'idx_dclose_ND1', 'stk_dclose_ND1_adj', 'stk_dv_ND1_adj']]
    return mta[['skey', 'interval', 'alp_1d',  'beta', 'idx_dv_ND1', 'stk_dv_ND1_adj']]
def calcDaily(date):
    pairs = []
    daily_data_day = daily_data.loc[daily_data['date'] == date]
    print(daily_data_day)
    for i in range(len(daily_data_day)):
        pairs.append((daily_data_day.iloc[i]['date'], daily_data_day.iloc[i]['skey']))
    
    P = mp.Pool(96)
    df_l = P.map(loadMbd, pairs)
    P.close()
    P.join()
    df = pd.concat(df_l)
    print(df)

    

before = datetime.now()     
if __name__ == '__main__':

    ##############################
    #set param list: IC 21.01-03 #
    ##############################
    startDate = 20210104
    endDate = 20210104
    reader = AShareReader()
    idx = 'CSI1000'
    daily_data = reader.Read_Stock_Daily('com_md_eq_cn', 'mdbar1d_jq', startDate, endDate, univ=[idx])
    #print(daily_data)
    #print(daily_data['skey'] >= 2000000)
    #daily_data = daily_data.loc[daily_data['skey'] >= 2000000]
    #loadMbd((20210104, 1600519)).to_csv('/data/home/jianw/OrderAnalysis/20220222_modelFitting/moutai.csv', index = False)
    #print(reader.Read_Stock_Daily('com_md_eq_cn', 'chnuniv_csirest', 20210728, 20210728))
    pairs = []
    print(daily_data)
    for i in range(len(daily_data)):
        pairs.append((daily_data.iloc[i]['date'], daily_data.iloc[i]['skey']))
    
    #calcDaily(20210107)
    #loadMbd((20210104, 1600519)).to_csv('/data/home/jianw/OrderAnalysis/20220222_modelFitting/moutai.csv', index = False)
    
    
    loadMbd((20210312, 2000011))

    #print(idx)  
    '''
    P = mp.Pool(96)
    #P.map(loadAlphaF90, list(set(daily_data['date'])))
    df_l = P.map(loadMbd, pairs)
    P.close()
    P.join()
    '''
    
    

    
    
    
after = datetime.now()
print(after - before)