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


def calPortion(date):
    
    ## load summary data
    df_l = []
    file_path = "/data/home/jianw/OrderAnalysis/20220303_mtaRealization/data/rawData/%s/%s/" %(idx, date.replace('-', ''))
    for file in os.listdir(file_path):
        df = pd.read_parquet(file_path + file)
        df_l.append(df)
    df = pd.concat(df_l)
    df = df.loc[df['interval']< 145800000000]
    #df = df.loc[(df['interval']> 144500000000) & (df['interval']<= 145600000000)]
    df['beta'] = df['beta'].fillna(1)

    sta = loadAlphaF90(date.replace('-', ''))
    #sta.to_parquet('/data/home/jianw/OrderAnalysis/20220303_mtaRealization/sta.parquet')
    df = df.merge(sta[['skey', 'ordering', 'yHatBuy90']], on = ['skey', 'ordering'], how = 'left')
    print(df)
    
    
    
    df['topBar'] = df.groupby('interval')['alp_1d'].transform(lambda x: np.quantile(x, 1-top))
    df_top = df.loc[df['alp_1d']>=df['topBar']]
    df_top['retTop'] = df_top.groupby('interval')['ret_30s'].transform(lambda x: np.quantile(x, 0.8))
    df_top['retBottom'] = df_top.groupby('interval')['ret_30s'].transform(lambda x: np.quantile(x, 0.2))
    df_top = df_top.loc[(df_top['ret_30s'] >= df_top['retBottom']) & (df_top['ret_30s'] <= df_top['retTop'])]
    #df_top.to_csv('/data/home/jianw/OrderAnalysis/20220222_modelFitting/mtaRealization/topName/%s/top%s_%s.csv' %(idx, top, date), index = False)
    
    '''
    q50_30s = np.quantile(df_top['ret_30s'], 0.50) * 10000
    q50_1d = np.quantile(df_top['ret_1d'], 0.50) * 10000
    q50_30s_1d = np.quantile(df_top['ret_30s_1d'], 0.50) * 10000
    '''
    

    mean_1d = df_top['ret_1d'].mean() * 10000
    mean_30s = df_top['ret_30s'].mean() * 10000
    mean_30s_1d = df_top['ret_30s_1d'].mean() * 10000
    mean_50s = df_top['ret_50s'].mean() * 10000
    mean_50s_1d = df_top['ret_50s_1d'].mean() * 10000
    
    
    #print(np.quantile(df_top['ret_1d'], 0.50) * 10000 , q50_top20_1d, df_top['ret_1d'].mean()*10000)
    
    #df = pd.DataFrame({'date': date, 'median_30s': q50_30s, 'mean_30s': mean_30s, 'median_1d': q50_1d, 'mean_1d': mean_1d, 'median_30s_1d': q50_30s_1d, 'mean_30s_1d': mean_30s_1d}, index = [0])
    df = pd.DataFrame({'date': date, 'mean_30s': mean_30s, 'mean_50s': mean_50s,  'mean_1d': mean_1d, 'mean_30s_1d': mean_30s_1d, 'mean_50s_1d': mean_50s_1d}, index = [0])
    return(df)
    

    

before = datetime.now()     
if __name__ == '__main__':
    RCH_DIR = '/data/rch/'
    data_path = '/home/jianw/monetization_research/rawData_jian/data_jian_20220119/if/'
    sdate = '2021-01-01'
    edate = '2021-11-29'
    
    Idxs = ['IF', 'IC', 'CSI1000']
    Tops = [0.05, 0.1, 0.2]
    idx = 'IF'
    top = 0.2
    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    calPortion('2021-01-04')
    
    
after = datetime.now()
print(after - before)