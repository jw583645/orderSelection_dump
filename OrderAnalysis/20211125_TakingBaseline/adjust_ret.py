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



def add_day(date, offset=0):

    date_list = pd.read_csv("/data/rch//raw/secData_TR/tradeDate.csv", header=0)  
    date_list = date_list['date']
    date_list = date_list.tolist()
    return (date_list[date_list.index(date) + offset])
    
    
def adj_dividend(date):
    price_today = pd.read_csv("/data/rch/raw/marketData_TR/dailyData/stock/2021/stock_%s.csv.gz" %date)
    price_tmr = pd.read_csv("/data/rch/raw/marketData_TR/dailyData/stock/2021/stock_%s.csv.gz" %add_day(date,1))
    
    df = pd.merge(price_today[['ID', 'close']], price_tmr[['ID', 'yclose']], how = 'left')
    df['ND_to_today'] = df['yclose']/df['close']
    df['ND_to_today'] = df['ND_to_today'].fillna(1)
    
    
    return(df.rename(columns = {'ID':'skey'}))

def adj_ret(date):

    df = pd.read_parquet("/data/home/jianw/mta_eq_cn/data/rch/eq/mret_fwd/CHNUniv.EqALL4/1MIN/1.0.1/mret_fwd.CHNUniv.EqALL4.1MIN.%s.parquet" %date)
    adj_div = adj_dividend(date)
    df = df.merge(adj_div, on = 'skey', how = 'left')
    df['ND_to_today'] = df['ND_to_today'].fillna(1)
    
    df['stk_dv_ND1_adj'] = df['stk_dv_ND1']/ df['ND_to_today']
    df['stk_dclose_ND1_adj'] = df['stk_dclose_ND1']/ df['ND_to_today']
   
    df.to_parquet('/data/home/jianw/monetization_research/adjDivPrice/mret_fwd.CHNUniv.EqALL4.1MIN.%s.parquet'%date)
    #df.to_parquet()




before = datetime.datetime.now()          
if __name__ == '__main__':
    
    top = 0.05
    order_cap = 1000000
    order_side = 1
    method = 'stock'

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/sta/data_jian_newagg/'
    sdate = '2021-01-04'
    edate = '2021-12-01'

    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    

    
    P = mp.Pool(96)
    df_list = P.map(adj_ret, dates)
    P.close()
    P.join()
    

    

    
    
    

after = datetime.datetime.now()
print(after - before)