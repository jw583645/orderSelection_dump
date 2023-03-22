import numpy as np
import pandas as pd
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import os, sys
os.environ["OMP_NUM_THREADS"] = '1'


def concat_stock(skey):
    df = pd.read_csv(skey)
    df = df.loc[df['order_side']==1]
    df = df[['skey', 'yHatBuy90','alp_1d', '1d_ret', 'trade_qty', 'trade_price', 'adjMidF90s']]
    #df['trade_amt'] = df['trade_qty'] * df['trade_price'] 
    return(df)
    
def plotSTA(date):
    print(date)
    skeys = os.listdir(data_path + re.sub("[^0-9]", "", date) +'/' )
    skeys = [data_path + re.sub("[^0-9]", "", date) + '/'  + skey for skey in skeys]
    DF_l = []
    for skey in skeys:
        #print(skey)
        DF_l.append(concat_stock(skey))
    DF = pd.concat(DF_l)
    
    DF['trade_amt'] = DF['trade_qty'] * DF['trade_price']
    DF['90s_ret'] = (DF['adjMidF90s'] - DF['trade_price']) / DF['trade_price']
    
    DF = DF.reset_index(drop = True)
    df = DF

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(25, 15)         
    plt.scatter(df['yHatBuy90'],df['90s_ret'],s = (df['trade_amt'])**(2/3)*0.02,  alpha=0.1)
    plt.xlabel('sta', fontsize = 25)
    plt.ylabel('90s_ret', fontsize = 25)
    corr = df[['yHatBuy90','90s_ret']].corr().iloc[0,1]
    plt.title('STA Prediction %s: corr = %s' %(date, round(corr,2)), fontsize = 30)
    plt.savefig('/data/home/jianw/OrderAnalysis/plot/STA/STA_%s.png' %date)
    plt.show()

def plotSTAMTA_distribution(date):
    df = pd.read_csv('%s/%s/%s_IS.csv'%(data_path, re.sub("[^0-9]", "", date), date) )
    df1 = df.loc[df['1d_ret']>0]
    df2 = df.loc[df['1d_ret']<=0]
    
    ret = Ret.loc[Ret['date']==date, 'buyRet'].values[0]

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 15)         
    #plt.scatter(df1['alp_1d'],df1['1d_ret'],s = (df1['trade_amt'])**(2/3)*0.01, color = 'red', alpha=0.01)
    plt.scatter(df2['alp_1d']*10000,df2['yHatBuy90']*10000,s = (df2['trade_amt'])**(2/3)*0.005,  c=-df2['1d_ret'], cmap='Blues', alpha=0.05)
    plt.scatter(df1['alp_1d']*10000,df1['yHatBuy90']*10000,s = (df1['trade_amt'])**(2/3)*0.005,  c=df1['1d_ret'], cmap='Reds', alpha=0.05)
    plt.xlabel('mta', fontsize = 25)
    plt.ylabel('sta', fontsize = 25)
    plt.title('MTA_STA Distribution_%s; buyRet = %s bps' %(date, round(ret, 1)), fontsize = 30)
    
    plt.axhline(y=0, color = 'black', linestyle='-', lw=0.5)
    plt.axvline(x=0, color = 'black', linestyle='-', lw=0.5)
    plt.grid(True, which='both', linestyle = '--', linewidth = 0.5)
    plt.axline((10,0),(0, 20), color = 'black', linestyle='-', lw=0.5)
    plt.grid(True, which='both', linestyle = '--', linewidth = 0.5)
    plt.savefig('/data/home/jianw/OrderAnalysis/plot/MTASTA_Distribution/MTASTA_Distribution_%s.png' %date)
    plt.show()

    
if __name__ == '__main__':

    data_path = '/data/home/jianw/OrderAnalysis/data/'
    RCH_DIR= '/data/rch/'
    sdate = '2021-01-01'
    edate = '2021-03-31'
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    Ret = pd.read_csv("/data/home/jianw/OrderAnalysis/linearCombine/data/0.5-10.csv")
    P = mp.Pool(96)
    P.map(plotSTAMTA_distribution, dates)
    P.close()
    P.join()