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


def realizedRet(tuples):
    IS, date, sta_wt, threshold, mta_multi, sta_multi = tuples
    Sample_dic = {1:'IS', 0: 'ALL', 2:'OOS'}
    print(tuples)
    df = pd.read_csv('%s/%s/%s_%s.csv'%(data_path, re.sub("[^0-9]", "", date), date, Sample_dic[IS]))
    df['trade_amt'] = np.minimum(df['trade_amt'], 1000000)
    #df = df.loc[df['trade_amt'] < 1000000]
    df = df.loc[(df['alp_1d'] + sta_wt * df['prev_yHatBuy90'] > threshold/10000)]
    #df = df.loc[(df['yHatBuy90'] >= 0)&(df['alp_1d'] < 0)]
    tradeAmt = df['trade_amt'].sum()
    profitAmt = (df['trade_amt'] * (df['1d_ret'] - 0.0013)).sum()      
    try:
        os.makedirs('/data/home/jianw/OrderAnalysis/linearCombine/data/%s-%s' %(sta_wt, threshold))
    except:
        pass
    df = pd.DataFrame({'date': date, 'tradeAmt': tradeAmt, 'profitAmt': profitAmt }, index = [0])
    df['buyRet'] = df['profitAmt'] / df['tradeAmt'] *10000
    #df.to_csv('/data/home/jianw/OrderAnalysis/linearCombine/data/%s-%s/%s.csv' %(sta_wt, threshold, date), index = False)
    return(df)
'''
def realizedRet(tuples):
    IS, date, sta_wt, threshold, mta_multi, sta_multi = tuples
    Sample_dic = {1:'IS', 0: 'ALL', 2:'OOS'}
    print(tuples)
    df = pd.read_csv('%s/%s/%s_%s.csv'%(data_path, re.sub("[^0-9]", "", date), date, Sample_dic[IS]))
    #df = df.loc[(df['alp_1d'] + sta_wt * df['yHatBuy90'] > threshold/10000)|(df['alp_1d']>mta_multi*threshold/10000)|(df['yHatBuy90']>sta_multi*threshold/10000)]
    df = df.loc[(df['alp_1d'] + sta_wt * df['yHatBuy90'] > threshold/10000)]
    #df = df.loc[(df['yHatBuy90'] >= 0)&(df['alp_1d'] < 0)]
    tradeAmt = df['trade_amt'].sum()
    profitAmt = (df['trade_amt'] * df['1d_ret']).sum()      
    try:
        os.makedirs('/data/home/jianw/OrderAnalysis/linearCombine/data/%s-%s' %(sta_wt, threshold))
    except:
        pass
    df = pd.DataFrame({'date': date, 'tradeAmt': tradeAmt, 'profitAmt': profitAmt - 0.0013*tradeAmt }, index = [0])
    df['buyRet'] = df['profitAmt'] / df['tradeAmt'] *10000
    #df.to_csv('/data/home/jianw/OrderAnalysis/linearCombine/data/%s-%s/%s.csv' %(sta_wt, threshold, date), index = False)
    return(df)
    #return(tradeAmt, profitAmt)
'''
before = datetime.datetime.now()          
if __name__ == '__main__':

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/OrderAnalysis/data/'
    sdate = '2021-01-01'
    edate = '2021-03-31'


    
    IS = 1
    sta_wts = [0.1, 0.5, 1]
    thresholds = [8, 10, 12, 14, 16]
    #sta_thresholds = [100, 1,1.5, 2, 2.5 ,3,3.5, 4, 4.5, 5]
    #mta_thresholds = [100, 1,1.5, 2, 2.5 ,3,3.5, 4, 4.5, 5]
    #mta_thresholds = [1,2,3,4,5, 100]
    #sta_thresholds = [1,2,3,4, 4.5, 5, 100]
    #mta_thresholds = [1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    #sta_thresholds = [5.1, 5.3,5.5, 5.7, 5.9, 6.1]
    mta_thresholds = 100
    sta_thresholds = 100
    mta_multi, sta_multi = 100, 100
    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    '''
    tuples = []
    for date in dates:
        tuples.append((IS, date, sta_wt, threshold, 0, 0))
    P = mp.Pool(96)
    DF_l = P.map(realizedRet, tuples)
    P.close()
    P.join()
            
    DF = pd.concat(DF_l)
    DF.to_csv('/data/home/jianw/OrderAnalysis/linearCombine/data/%s-%s_new.csv' %(sta_wt, threshold), index = False)
    '''
    
    Profit = []
    TradingAmt = []
    Ret = []
    for sta_wt in sta_wts:
        Profit_sta = []
        TradingAmt_sta = []
        Ret_sta = []
        for threshold in thresholds:            
            tuples = []
            for date in dates:
                tuples.append((IS, date, sta_wt, threshold, mta_multi, sta_multi))
            P = mp.Pool(96)
            DF_l = P.map(realizedRet, tuples)
            P.close()
            P.join()
            
            DF = pd.concat(DF_l)
            profit = DF['profitAmt'].mean()
            Profit_sta.append(profit)
            tradingAmt = DF['tradeAmt'].mean()
            TradingAmt_sta.append(tradingAmt)
            ret = profit/tradingAmt * 10000
            Ret_sta.append(ret)
            DF.to_csv('/data/home/jianw/OrderAnalysis/linearCombine/data/%s-%s.csv' %(sta_wt, threshold), index = False)
        Profit.append(Profit_sta)
        TradingAmt.append(TradingAmt_sta)
        Ret.append(Ret_sta)
    Profit = pd.DataFrame(Profit, columns = thresholds, index = sta_wts)
    TradingAmt = pd.DataFrame(TradingAmt, columns = thresholds, index = sta_wts)
    Ret = pd.DataFrame(Ret, columns = thresholds, index = sta_wts)
    print(Profit)
    print(Ret)
    Profit.to_csv('/data/home/jianw/OrderAnalysis/linearCombine/data/Profit.csv')
    TradingAmt.to_csv('/data/home/jianw/OrderAnalysis/linearCombine/data/TradingAmt.csv')
    Ret.to_csv('/data/home/jianw/OrderAnalysis/linearCombine/data/Ret.csv')
    

after = datetime.datetime.now()
print(after - before)