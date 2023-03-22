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
#import datetime
#from datetime import datetime
os.environ["OMP_NUM_THREADS"] = '1'

def calforward30sRet(date):
    #if os.path.exists('/data/home/jianw/OrderAnalysis/20220222_modelFitting/mtaRealization/if/%s.parquet' %date):
       #return None
    df = pd.read_parquet('/data/home/jianw/monetization_research/rawData_jian/concat_data/IF/%s.parquet'%date)
    df = df.sort_values(['skey', 'time'])
    df = df.reset_index(drop = True)
    #df['minute'] = orange_interval(df['time'])
    df['safeBid1p'] = np.where(df.eval('bid1p==0'), df['ask1p'], df['bid1p'])
    df['safeAsk1p'] = np.where(df.eval('ask1p==0'), df['bid1p'], df['ask1p'])
    #df['adjMid'] = df.eval('(safeBid1p * ask1q + safeAsk1p * bid1q) / (bid1q + ask1q)')
    df['adjMid'] = (df['safeBid1p'] + df['safeAsk1p']) / 2
    
    df['minute_30s'] = df['interval'] + 30000000
    df['minute_50s'] = df['interval'] + 50000000
    
    df['skey_time'] = list(zip(df['skey'], df['time']))
    df = df.reset_index()
    df['skey_minute_30s'] = list(zip(df['skey'], df['minute_30s']))
    df['skey_minute_50s'] = list(zip(df['skey'], df['minute_50s']))
    df['skey_minute_orange'] = list(zip(df['skey'], df['interval']))
    
    new_idx0 = np.minimum(np.searchsorted(df['skey_time'], df['skey_minute_orange']), df.groupby(['skey'])['index'].transform('max'))
    df['if_orange'] = df.iloc[new_idx0]['if'].reset_index(drop = True)
    df['adjMid_orange'] = df.iloc[new_idx0]['adjMid'].reset_index(drop = True)
    
    new_idx1 = np.minimum(np.searchsorted(df['skey_time'], df['skey_minute_30s']), df.groupby(['skey'])['index'].transform('max'))
    df['if_30s'] = df.iloc[new_idx1]['if'].reset_index(drop = True)
    df['adjMid_30s'] = df.iloc[new_idx1]['adjMid'].reset_index(drop = True)
    
    new_idx2 = np.minimum(np.searchsorted(df['skey_time'], df['skey_minute_50s']), df.groupby(['skey'])['index'].transform('max'))
    df['if_50s'] = df.iloc[new_idx2]['if'].reset_index(drop = True)
    df['adjMid_50s'] = df.iloc[new_idx2]['adjMid'].reset_index(drop = True)
    
    df = df.drop(columns=['skey_time', 'skey_minute_30s', 'skey_minute_50s', 'skey_minute_orange'] )
    
    ## calculate realization
    #df = df.loc[df['order_side'] == 1]
    df = df.groupby(['skey', 'interval']).agg({'trade_qty_taking': 'sum', 'alp_1d': 'mean', 'if_orange' : 'mean', 'adjMid_orange': 'mean', 'if_30s': 'mean',
                                      'adjMid_30s': 'mean', 'if_50s': 'mean', 'adjMid_50s': 'mean', 'beta': 'mean' }).reset_index()
    
    df['ret_30s'] = np.log(df['adjMid_30s']/df['adjMid_orange']) - df['beta'] * np.log(df['if_30s']/df['if_orange'])
    df['ret_50s'] = np.log(df['adjMid_50s']/df['adjMid_orange']) - df['beta'] * np.log(df['if_50s']/df['if_orange'])
    #df['mtaRealization'] = df['ret_30s'] / df['alp_1d']
    #df['mtaRealization_30s'] = df['ret_30s']
    
    df.to_parquet('/data/home/jianw/OrderAnalysis/20220222_modelFitting/mtaRealization/IF/%s.parquet' %date)
    return(df)

def calPortion(date):
    df = pd.read_parquet('/data/home/jianw/OrderAnalysis/20220222_modelFitting/mtaRealization/CSI1000/%s.parquet' %date)
    df = df.loc[~df['ret_30s'].isna()]
    df = df.loc[~df['ret_50s'].isna()]
    #q25 = np.quantile(df['mtaRealization'], 0.25)
    #q50 = np.quantile(df['mtaRealization'], 0.50)
    #q75= np.quantile(df['mtaRealization'], 0.75)
    df['top20Bar'] = df.groupby('interval')['alp_1d'].transform(lambda x: np.quantile(x, 0.8))
    df['top10Bar'] = df.groupby('interval')['alp_1d'].transform(lambda x: np.quantile(x, 0.9))
    df['top5Bar'] = df.groupby('interval')['alp_1d'].transform(lambda x: np.quantile(x, 0.95))
    
    
    df_top20 = df.loc[df['alp_1d']>df['top20Bar']]
    q25_top20_30s = np.quantile(df_top20['ret_30s'], 0.25)
    q50_top20_30s = np.quantile(df_top20['ret_30s'], 0.50)
    q75_top20_30s = np.quantile(df_top20['ret_30s'], 0.75)
        
    df_top10 = df.loc[df['alp_1d']>df['top10Bar']]
    q25_top10_30s = np.quantile(df_top10['ret_30s'], 0.25)
    q50_top10_30s = np.quantile(df_top10['ret_30s'], 0.50)
    q75_top10_30s = np.quantile(df_top10['ret_30s'], 0.75)
        
    df_top5 = df.loc[df['alp_1d']>df['top5Bar']]
    q25_top5_30s = np.quantile(df_top5['ret_30s'], 0.25)
    q50_top5_30s = np.quantile(df_top5['ret_30s'], 0.50)
    q75_top5_30s = np.quantile(df_top5['ret_30s'], 0.75)
    
    
    ## 50s
    q25_top20_50s = np.quantile(df_top20['ret_50s'], 0.25)
    q50_top20_50s = np.quantile(df_top20['ret_50s'], 0.50)
    q75_top20_50s = np.quantile(df_top20['ret_50s'], 0.75)
            
    q25_top10_50s = np.quantile(df_top10['ret_50s'], 0.25)
    q50_top10_50s = np.quantile(df_top10['ret_50s'], 0.50)
    q75_top10_50s = np.quantile(df_top10['ret_50s'], 0.75)
            
    q25_top5_50s = np.quantile(df_top5['ret_50s'], 0.25)
    q50_top5_50s = np.quantile(df_top5['ret_50s'], 0.50)
    q75_top5_50s = np.quantile(df_top5['ret_50s'], 0.75)
    
    
    

    
    df = pd.DataFrame({'date': date,  'q25_top20_30s': q25_top20_30s, 'q50_top20_30s': q50_top20_30s, 'q75_top20_30s': q75_top20_30s, 'q25_top10_30s': q25_top10_30s, 'q50_top10_30s': q50_top10_30s, 'q75_top10_30s': q75_top10_30s, 
    'q25_top5_30s': q25_top5_30s, 'q50_top5_30s': q50_top5_30s, 'q75_top5_30s': q75_top5_30s, 'q25_top20_50s': q25_top20_50s, 'q50_top20_50s': q50_top20_50s, 'q75_top20_50s': q75_top20_50s, 'q25_top10_50s': q25_top10_50s, 'q50_top10_50s': q50_top10_50s, 'q75_top10_50s': q75_top10_50s, 
    'q25_top5_50s': q25_top5_50s, 'q50_top5_50s': q50_top5_50s, 'q75_top5_50s': q75_top5_50s}, index = [0])
    
    return(df)
    
    
    
    

before = datetime.now()          
if __name__ == '__main__':
    

    RCH_DIR = '/home/jianw/rch/'
    data_path = '/home/jianw/monetization_research/rawData_jian/data_jian_20220119/if/'
    sdate = '2021-01-01'
    edate = '2021-11-29'

    
    #date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    #date_list = date_list['date']
    #date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    #dates = date_list.to_list()
    #concat_stock('2021-04-23')
    
    date_list = pd.read_csv("/data/home/jianw/OrderAnalysis/ddates.csv")
    date_list = date_list['0'].astype(str)
    
    
    P = mp.Pool(96)
    df_list = P.map(calPortion, date_list)
    #P.map(calforward30sRet, date_list)
    P.close()
    P.join()
    
    df = pd.concat(df_list)
    #print(pd.DataFrame(df.mean()))
    #df.append(df.mean())
    #print(df)
    df.to_csv('/data/home/jianw/OrderAnalysis/20220222_modelFitting/mtaRealization/CSI1000.csv', index = False)
    
    
    
    
    #read_mta('2021-01-25')
    
    

after = datetime.now()
print(after - before)