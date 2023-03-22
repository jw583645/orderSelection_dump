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


def plotSTAMTA_distribution(date):
    print(date)
    df = pd.read_csv('%s/%s/%s_IS.csv'%(data_path, re.sub("[^0-9]", "", date), date) )
    df['prev_yHatBuy90']*=10000
    df['prev_yHatBuy90'] = round(df['prev_yHatBuy90'],0)
    df['alp_1d']*=10000
    df['alp_1d'] = round(df['alp_1d'],0)
    df['netPnL'] = df['trade_amt'] * (df['1d_ret']- 0.0013)
    df = df[['prev_yHatBuy90', 'alp_1d', 'trade_amt', 'netPnL']]
    df = df.groupby(['alp_1d', 'prev_yHatBuy90'], as_index = False).agg({'trade_amt':'sum', 'netPnL':'sum'})
    return(df)

    
if __name__ == '__main__':

    data_path = '/data/home/jianw/OrderAnalysis/data/'
    RCH_DIR= '/data/rch/'
    sdate = '2021-01-01'
    edate = '2021-03-31'
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    P = mp.Pool(96)
    DF_l = P.map(plotSTAMTA_distribution, dates)
    P.close()
    P.join()
    df = pd.concat(DF_l)
    df = df.groupby(['alp_1d', 'prev_yHatBuy90'], as_index = False).agg({'trade_amt':'sum', 'netPnL':'sum'})
    df['netRet'] = df['netPnL']/df['trade_amt']*10000
    
    
    trade_all = df['trade_amt'].sum()
    Portion = []
    for r in [10, 20, 30,40,50,60,70, 80, 90, 100, 150, 200, 250, 300]:
        Portion.append(df.loc[df['alp_1d']**2 + 2* df['prev_yHatBuy90']**2 < r**2 ]['trade_amt'].sum()/trade_all)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 15)         
    plt.plot([10, 20, 30,40,50,60,70, 80, 90, 100, 150, 200, 250, 300],Portion)
    plt.xlabel('radius', fontsize = 25)
    plt.ylabel('portion', fontsize = 25)
    plt.title('Trading_Amt_percentage', fontsize = 30)
    plt.grid(True, which='both', linestyle = '--', linewidth = 0.5)
    plt.savefig('/data/home/jianw/OrderAnalysis/plot/MTASTA_Distribution/Trading_Amt_percentage.png')
    plt.show()
    '''
    #print(df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 10**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 20**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 30**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 40**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 50**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 60**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 70**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 80**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 90**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 100**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 150**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    print(df.loc[df['alp_1d']**2 + df['alp_1d']**2 < 200**2 ]['trade_amt'].sum()/df['trade_amt'].sum())
    '''
    ##########################
    df1 = df.loc[df['netRet']>0]
    df2 = df.loc[df['netRet']<=0]

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(15, 15)         
    #plt.scatter(df1['alp_1d'],df1['1d_ret'],s = (df1['trade_amt'])**(2/3)*0.01, color = 'red', alpha=0.01)
    #plt.scatter(df2['alp_1d'],df2['prev_yHatBuy90'],s = (df2['trade_amt'])**(2/3)*0.0001,  c=-df2['netRet'], cmap='Blues', alpha=0.5)
    #plt.scatter(df1['alp_1d'],df1['prev_yHatBuy90'],s = (df1['trade_amt'])**(2/3)*0.0001,  c=df1['netRet'], cmap='Reds', alpha=0.5)
    #plt.scatter(df2['alp_1d'],df2['prev_yHatBuy90'],s =0.5,  c=(abs(df2['netPnL'])**(2/3)), cmap='Blues', alpha=1)
    #plt.scatter(df1['alp_1d'],df1['prev_yHatBuy90'],s =0.5,  c=df1['netPnL']**(2/3), cmap='Reds', alpha=1)
    
    plt.scatter(df['alp_1d'],df['prev_yHatBuy90'],s =15,  c=(abs(df['trade_amt'])**(1/2)), cmap='Blues', alpha=1)
    plt.xlim(-100, 100) 
    plt.ylim(-100, 100) 
    
    
    plt.xlabel('mta', fontsize = 25)
    plt.ylabel('sta', fontsize = 25)
    plt.title('MTA_STA Distribution_trade_amt$', fontsize = 30)
    
    plt.axhline(y=0, color = 'black', linestyle='-', lw=0.5)
    plt.axvline(x=0, color = 'black', linestyle='-', lw=0.5)
    plt.grid(True, which='both', linestyle = '--', linewidth = 0.5)
    #plt.axline((10,0),(0, 20), color = 'black', linestyle='-', lw=0.5)
    plt.savefig('/data/home/jianw/OrderAnalysis/plot/MTASTA_Distribution/MTASTA_Distribution_tradeAmt_agg.png')
    plt.show()
    