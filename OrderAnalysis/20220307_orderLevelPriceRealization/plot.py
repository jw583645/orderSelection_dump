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
import datetime

def makePlot(idx):
    df = pd.read_csv('/data/home/jianw/OrderAnalysis/20220307_orderLevelPriceRealization/data/summaryData/%s_summary.%s.2021-01-01.2021-11-29.csv' %(idx,spec))
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 12)  
    #df['time'] = pd.to_datetime(df['time'])
    #df['time'] = df['time'].dt.time
    df['date'] = pd.to_datetime(df['date'])
    plt.plot(df['date'], df['trade_amt']/1000000000, label = 'trade_amt: %s yi' %round(df['trade_amt'].mean()/1e9,1))
    
    plt.legend(fontsize = 20)
    plt.ylabel('yi', fontsize = 20)
    plt.title('%s Trade Amt_top 20%%_%s' %(idx,spec), fontsize = 25)
    plt.savefig('/data/home/jianw/OrderAnalysis/20220307_orderLevelPriceRealization/plot/%s.tradeAmt.%s.png'%(idx,spec))
    plt.close()
    
    ## tradeAmt
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 12)  
    #df['time'] = pd.to_datetime(df['time'])
    #df['time'] = df['time'].dt.time
    df['date'] = pd.to_datetime(df['date'])
    Bars = [0,1,2,3,4,5,6,7]
    bars_dic = {0:'<-100',1:'-100~-50',2:'-50~-20',3: '-20~0',4: '0~20',5:'20~50',6: '50~100',7:'>100'}
    for bar in Bars:
        plt.plot(df['date'], df['trade_amt_%s' %bar], label = '%s portion: %s%%' %(bars_dic[bar], round(df['trade_amt_%s' %bar].mean()*100,1)))
    
    plt.legend(fontsize = 20)
    #plt.ylabel('yi', fontsize = 20)
    plt.title('%s Trade Amt Portion_%s' %(idx,spec), fontsize = 25)
    plt.savefig('/data/home/jianw/OrderAnalysis/20220307_orderLevelPriceRealization/plot/%s.tradePortion.%s.png'%(idx,spec))
    plt.close()
    
    ## 1d return
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 12)  
    #df['time'] = pd.to_datetime(df['time'])
    #df['time'] = df['time'].dt.time
    df['date'] = pd.to_datetime(df['date'])
    Bars = [0,1,2,3,4,5,6,7]
    bars_dic = {0:'<-100',1:'-100~-50',2:'-50~-20',3: '-20~0',4: '0~20',5:'20~50',6: '50~100',7:'>100'}
    for bar in Bars:
        plt.plot(df['date'], df['ret_%s' %bar].rolling(window=22).mean(), label = '%s trade_1d_ret: %s' %(bars_dic[bar], round(df['ret_%s' %bar].mean()*10000,1)))
    
    plt.legend(fontsize = 20)
    #plt.ylabel('yi', fontsize = 20)
    plt.title('%s trade_1d Ret_rolling1M_%s'%(idx,spec), fontsize = 25)
    plt.savefig('/data/home/jianw/OrderAnalysis/20220307_orderLevelPriceRealization/plot/%s.Ret.%s.png'%(idx,spec))
    plt.close()
    
    ## alp_1d
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 12)  
    #df['time'] = pd.to_datetime(df['time'])
    #df['time'] = df['time'].dt.time
    df['date'] = pd.to_datetime(df['date'])
    Bars = [0,1,2,3,4,5,6,7]
    bars_dic = {0:'<-100',1:'-100~-50',2:'-50~-20',3: '-20~0',4: '0~20',5:'20~50',6: '50~100',7:'>100'}
    for bar in Bars:
        plt.plot(df['date'], df['alp_1d_%s' %bar].rolling(window=22).mean(), label = '%s alp_1d: %s' %(bars_dic[bar], round(df['alp_1d_%s' %bar].mean()*10000,1)))
    
    plt.legend(fontsize = 20)
    #plt.ylabel('yi', fontsize = 20)
    plt.title('%s y_hat_1d_rolling1M_%s'%(idx,spec), fontsize = 25)
    plt.savefig('/data/home/jianw/OrderAnalysis/20220307_orderLevelPriceRealization/plot/%s.Alpha1d.%s.png'%(idx,spec))
    plt.close()
    

spec = 'openning30'  
Idxs = ['IF', 'IC', 'CSI1000', 'CSIRest']
for idx in Idxs:
    makePlot(idx)