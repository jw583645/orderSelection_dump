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


def concat_stock(tuples):
    IS, date = tuples
    Sample_dic = {1:'IS', 0: 'ALL', 2:'OOS'}
    '''
    try:
        os.remove('%s/%s/%s_%s.csv'%(data_path, re.sub("[^0-9]", "", date), date, Sample_dic[IS]))
        #os.remove('%s/%s/%s_%s.csv.gz'%(data_path, re.sub("[^0-9]", "", date), date, Sample_dic[IS]))
    except:
        pass
    '''
    skeys = os.listdir(data_path + re.sub("[^0-9]", "", date) +'/' )
    skeys = [val for val in skeys if not val.endswith(".csv")]
    skeys = set(np.array([i[:7] for i in skeys]).astype(int))
    if IS == 1:
        skeys_IS = pd.read_csv("/data/home/jianw/OrderAnalysis/IS_Stocks/stocksIS_%s.csv" %date)
        skeys_IS = set(skeys_IS['stocksIS'])
        skeys = skeys.intersection(skeys_IS)
    elif IS == 2:
        skeys_IS = pd.read_csv("/data/home/jianw/OrderAnalysis/IS_Stocks/stocksIS_%s.csv" %date)
        skeys_IS = set(skeys_IS['stocksIS'])
        skeys = skeys - skeys_IS
    DF_l = []
    
    for skey in skeys:
        #df = pd.read_csv( '%s/%s/%s.csv.gz'%(data_path, re.sub("[^0-9]", "", date), skey) )
        df = pd.read_csv( '%s/%s/%s.csv.gz'%(data_path, re.sub("[^0-9]", "", date), skey) )
        df = df.loc[df['order_side']==1]
        #df = df[['skey', 'prev_yHatBuy90','alp_1d', '1d_ret', 'trade_qty', 'trade_price', 'adjMidF90s', 'amountThisUpdate']]
        DF_l.append(df)
    DF = pd.concat(DF_l)
    
    #DF['trade_amt'] = DF['trade_qty'] * DF['trade_price']
    DF['trade_amt'] = DF['amountThisUpdate']
    DF['90s_ret'] = (DF['adjMidF90s'] - DF['trade_price']) / DF['trade_price']   
    df = DF.reset_index(drop = True)
    df.to_csv('%s/%s/%s_%s.csv'%(data_path, re.sub("[^0-9]", "", date), date, Sample_dic[IS]), index = False)
    print(df)
    #return(df)

before = datetime.datetime.now()          
if __name__ == '__main__':

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/OrderAnalysis/data_feature/'
    sdate = '2021-01-01'
    edate = '2021-03-31'
    

    
    IS = 2
    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
             
    tuples = []
    for date in dates:
        tuples.append((IS, date))
    P = mp.Pool(96)
    DF_l = P.map(concat_stock, tuples)
    P.close()
    P.join()

after = datetime.datetime.now()
print(after - before)