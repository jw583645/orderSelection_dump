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
os.environ["OMP_NUM_THREADS"] = '1'
pd.set_option("display.max_columns", None)
from datetime import date
from datetime import datetime

def makePlot(sta_wt, univ):
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 7.5) 
    
    for wt_priceRealization in wts_priceRealization:
        df = pd.read_csv("/data/home/jianw/OrderAnalysis/20220331_Baseline/summary_mid/%s/new_top0.1.minute.minute.orderside%s.0-%s-%s.2021-01-01.2021-10-29.ALL.csv" %(univ,order_side, sta_wt, wt_priceRealization ))
        df_bps = df['return_bps'].iloc[-1]
        df = df.iloc[:-1]
        df['date'] = pd.to_datetime(df['date'])

        plt.plot(df['date'], (df['return_bps']).cumsum(), label = '%s: %s'%(wt_priceRealization,(round(df_bps,2))))
        
    plt.legend(fontsize = 20)
    #plt.xlabel('date', fontsize = 20)
    plt.ylabel('bps', fontsize = 20)
    #plt.ylim(0,)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=14)
    plt.xticks(rotation=30)
    plt.title('%s buy top 10%% cumRet' %univ, fontsize = 25)
    plt.savefig('/data/home/jianw/OrderAnalysis/20220331_Baseline/plot/%s.staWt%s.png' %(univ,sta_wt))
    plt.close()

def makePlotDiff(univ):
    df1 = pd.read_csv('/data/home/jianw/OrderAnalysis/20220331_Baseline/summary_mid/%s/new_top0.1.minute.minute.orderside1.0-0.5-0.2021-01-01.2021-10-29.ALL.csv'%univ)
    df2 = pd.read_csv('/data/home/jianw/OrderAnalysis/20220331_Baseline/summary_mid/%s/new_top0.1.minute.minute.orderside1.0-0.5-0.2.2021-01-01.2021-10-29.ALL.csv'%univ)
    #orderR = df1.iloc[-1]['order_amt_top'] / df.iloc[-1]['order_amt_all']
    impr = df2.iloc[-1]['return_bps'] - df1.iloc[-1]['return_bps']
    ret1 = df1.iloc[-1]['return_bps']
    ret2 = df2.iloc[-1]['return_bps']
                        
    df1 = df1.iloc[:-1]
    df2 = df2.iloc[:-1]
    
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 7.5)  
    
    
    sr = (df2['return_bps'] - df1['return_bps']).mean() / (df2['return_bps'] - df1['return_bps']).std() * (244**0.5)
    plt.plot(df1['date'], (df2['return_bps'] - df1['return_bps']).cumsum() , label = 'impv: %s, SR: %s'%(round(impr,2), round(sr, 2)) )
    #plt.plot(df1['date'], (df2['return_bps'] - df1['return_bps']).cumsum() )
    plt.legend(fontsize = 20)
    #plt.xlabel('date', fontsize = 20)
    plt.ylabel('bps', fontsize = 20)
    #plt.ylim(0,)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=14)
    plt.xticks(rotation=30)
    plt.title('%s Impv cumRet'%univ, fontsize = 25)
    plt.savefig('/data/home/jianw/OrderAnalysis/20220331_Baseline/plot/%s.impv.png' %(univ))
    plt.close()  


def makePlotSummary(univ):
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 12) 
    
    for sta_wt in sta_wts:
        rets = []
        for wt_priceRealization in wts_priceRealization:
            df = pd.read_csv("/data/home/jianw/OrderAnalysis/20220331_Baseline/summary_mid/%s/new_top0.1.minute.minute.orderside%s.0-%s-%s.2021-01-01.2021-10-29.ALL.csv" %(univ,order_side, sta_wt, wt_priceRealization ))
            df_bps = df['return_bps'].iloc[-1]
            rets.append(df_bps)
   
        plt.plot(wts_priceRealization, rets, label = 'staWt = %s'%sta_wt)
        
    plt.legend(fontsize = 20)
    plt.ylabel('bps', fontsize = 20)
    plt.xlabel('coef of priceRealization', fontsize = 20)
    plt.title('%s buyside top 10%% Optimal coef' %univ, fontsize = 25)
    plt.savefig('/data/home/jianw/OrderAnalysis/20220331_Baseline/plot/%s.order_side%s.optimalCoef.png' %(univ, order_side))
    plt.close()
  
before = datetime.now()     
if __name__ == '__main__':

    Univs = ['IF', 'IC', 'CSI1000']
    order_side = 1
    sta_wts = [0.0001, 1, 1.5,2]
    wts_priceRealization = [0, 0.1, 0.2, 0.3, 0.4]
    
    '''
    for sta_wt in sta_wts:
        for univ in Univs:  
            makePlot(sta_wt, univ)
    '''
    
    '''
    for univ in Univs:  
        makePlotDiff(univ)
    '''
    
    for univ in Univs:  
        makePlotSummary(univ)
    
    
after = datetime.now()
print(after - before)