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

def makePlot(sta_wt):
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 12) 
    
    for wt_priceRealization in wts_priceRealization:
        df = pd.read_csv("/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/summary/%s/top.minute.minute.orderside%s.0-%s-%s.2021-01-01.2021-10-29.ALL.csv" %(univ,order_side, sta_wt, wt_priceRealization ))
        df_bps = df['return_bps'].iloc[-1]
        df = df.iloc[:-1]
        df['date'] = pd.to_datetime(df['date'])

        plt.plot(df['date'], (df['return_bps']).cumsum(), label = '0.5-%s: %s'%(wt_priceRealization,(round(df_bps,2))))
        
    plt.legend(fontsize = 20)
    plt.ylabel('bps', fontsize = 20)
    plt.title('%s top 10%%' %univ, fontsize = 25)
    plt.savefig('/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/plot/%s.staWt%s.png' %(univ,sta_wt))
    plt.close()

def makePlotSummary():
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 12) 
    
    for sta_wt in sta_wts:
        rets = []
        for wt_priceRealization in wts_priceRealization:
            df = pd.read_csv("/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/summary/%s/top0.2.minute.minute.orderside%s.0-%s-%s.2021-01-01.2021-10-29.ALL.csv" %(univ,order_side, sta_wt, wt_priceRealization ))
            df_bps = df['return_bps'].iloc[-1]
            rets.append(df_bps)
   
        plt.plot(wts_priceRealization, rets, label = 'staWt = %s'%sta_wt)
        
    plt.legend(fontsize = 20)
    plt.ylabel('bps', fontsize = 20)
    plt.xlabel('coef of priceRealization', fontsize = 20)
    plt.title('%s sellside top 20%% Optimal coef' %univ, fontsize = 25)
    plt.savefig('/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/plot/%s.order_side%s.optimalCoef.png' %(univ, order_side))
    plt.close()
    
before = datetime.now()     
if __name__ == '__main__':

    univ = 'CSI1000'
    order_side = 2
    sta_wts = [0.0001, 0.5, 1, 1.5,2]
    wts_priceRealization = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    '''
    for sta_wt in sta_wts:
        makePlot(sta_wt)
    '''
    makePlotSummary()
    
    
after = datetime.now()
print(after - before)