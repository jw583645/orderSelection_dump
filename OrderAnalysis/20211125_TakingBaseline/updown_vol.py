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


def calc_mret_vol(date):
    df = pd.read_parquet("/data/home/jianw/mta_eq_cn/data/rch/eq/mret_fwd/CHNUniv.EqALL4/1MIN/1.0.1/mret_fwd.CHNUniv.EqALL4.1MIN.%s.parquet" %date)
    df['1d_ret'] = (df['stk_dv_ND1'] - df['mid']) / df['mid']
    
    print(df)
    df_up = df.loc[df['1d_ret']>=0]
    df_down = df.loc[df['1d_ret']<0]
    
    summary = pd.DataFrame([[date, df_up['1d_ret'].std(), df_down['1d_ret'].std()]],
                           columns=['date','up_vol', 'down_vol'])
    return(summary)

















before = datetime.datetime.now()          
if __name__ == '__main__':
    
    top = 0.05
    order_cap = 1000000
    order_side = 1
    method = 'stock'

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/sta/data_jian_newagg/'
    sdate = '2021-01-01'
    edate = '2021-06-30'

    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    

    
    P = mp.Pool(96)
    df_list = P.map(calc_mret_vol, dates)
    P.close()
    P.join()
    
    df = pd.concat(df_list)
    print(df)
    df.to_csv('/data/home/jianw/OrderAnalysis/20211125_TakingBaseline/updown_vol.csv', index = False)
    

    
    
    

after = datetime.datetime.now()
print(after - before)