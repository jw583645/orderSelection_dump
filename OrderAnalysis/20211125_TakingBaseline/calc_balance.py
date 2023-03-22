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


def calc_taking_making(date):
    print(date)
    skeys = os.listdir(data_path + re.sub("[^0-9]", "", date) +'/' )
    skeys = [val for val in skeys if not val.endswith(".csv")]
    skeys = set(np.array([i[:7] for i in skeys]).astype(int))
    DF_l = []
    
    for skey in skeys:
        df = pd.read_parquet( '%s/%s/%s.parquet'%(data_path, re.sub("[^0-9]", "", date), skey) )
        buy_taking = df.loc[(df['order_side'] == 1), 'trade_amt_taking'].sum()
        buy_making = df.loc[(df['order_side'] == 1), 'trade_amt_making'].sum()
        
        sell_taking = df.loc[(df['order_side'] == 2), 'trade_amt_taking'].sum()
        sell_making = df.loc[(df['order_side'] == 2), 'trade_amt_making'].sum()
        
        df = pd.DataFrame({'date': date, 'skey': skey, 'buy_taking': buy_taking, 'buy_making': buy_making, 'sell_taking': sell_taking, 'sell_making': sell_making}, index = [0])
        DF_l.append(df)
    DF = pd.concat(DF_l)
    print(DF)
    DF.to_csv('/data/home/jianw/OrderAnalysis/20211125_TakingBaseline/taking_making_balance/taking_making_balance.%s.csv'%date, index = False)
    
    


before = datetime.datetime.now()          
if __name__ == '__main__':
    

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/monetization_research/rawData_jian/data_jian_20211213/'
    sdate = '2021-01-01'
    edate = '2021-06-30'

    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    P = mp.Pool(96)
    P.map(calc_taking_making, dates)
    P.close()
    P.join()
    
    

after = datetime.datetime.now()
print(after - before)