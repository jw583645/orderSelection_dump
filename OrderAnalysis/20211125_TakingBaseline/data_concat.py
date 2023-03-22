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


def orange_interval(time):
    import datetime
    x = (time/1e6).astype(int).astype(str)
    y = np.where(x.str.len() < 6 , '0' + x.str.slice(0,1) + ':' + x.str.slice(1,3) + ':' + x.str.slice(3,5),
        x.str.slice(0,2) + ':' + x.str.slice(2,4) + ':' + x.str.slice(4,6))    
    z = pd.Series(((pd.to_datetime(y) - datetime.timedelta(seconds=20)).time).astype(str))
    interval = (z.str.replace(':','').astype(int)/100).astype(int)*1e8
    return(interval)
    
def read_mta(date):
    mta =  pd.read_parquet('/data/home/jianw/monetization_research/adjDivPrice/mret_fwd.CHNUniv.EqALL4.1MIN.%s.parquet' %(date))
    ic_dclose_ND1 = mta.loc[(mta['skey'] == 'SZ000009') & (mta['interval'] == '09:31:00'), 'idx_dclose_ND1'].iloc[0]
    #print(ic_dclose_ND1)
    mta = mta.loc[(mta['idx_dclose_ND1'] <= ic_dclose_ND1 + 1e-6)&(mta['idx_dclose_ND1'] >= ic_dclose_ND1 - 1e-6),]
    #mta = mta.loc[((mta['idx_dclose_ND1'] <= ic_dclose_ND1 + 1e-6)&(mta['idx_dclose_ND1'] >= ic_dclose_ND1 - 1e-6))|(mta['skey'] == 'SZ300750'),]
    #print(mta.loc[mta['skey']=='SZ300750'])
    print(date, len(list(set(mta['skey']))))
    mta['interval'] = (mta['interval'].str.replace(':', '')).astype(int)*1e6
    mta['skey'] = np.where(mta['skey'].str.slice(0,2) == 'SH', '1' + mta['skey'].str.slice(2), '2' + mta['skey'].str.slice(2)).astype(int)
    return mta[['skey', 'interval', 'alp_1d', 'alp_2d', 'beta', 'idx_dv_ND1', 'stk_dv_ND1', 'idx_N30min', 'stk_N30min', 'stk_dclose_ND1', 'idx_dv_ND2', 'stk_dv_ND2', 'idx_dclose_ND1', 'stk_dclose_ND1_adj', 'stk_dv_ND1_adj']]
    
    
    
def concat_stock(date):
    skeys = os.listdir(data_path + re.sub("[^0-9]", "", date) +'/' )
    skeys = [val for val in skeys if not val.endswith(".csv")]
    skeys = set(np.array([i[:7] for i in skeys]).astype(int))
    DF_l = []
    
    for skey in skeys:
        df = pd.read_parquet( '%s/%s/%s.parquet'%(data_path, re.sub("[^0-9]", "", date), skey) )
        df = df.loc[(df['order_type']==2) & (df['is_agg_intention']==1)]
        #df = df[['skey', 'prev_yHatBuy90','alp_1d', '1d_ret', 'trade_qty', 'trade_price', 'adjMidF90s', 'amountThisUpdate']]
        DF_l.append(df)
    DF = pd.concat(DF_l)
    
    #mta_rank = pd.read_csv("/data/home/jianw/sta/ICTopNames/ICTopButtom_%s.csv" %re.sub("[^0-9]", "", date))
    
    
    
    #DF['trade_amt'] = DF['trade_qty'] * DF['trade_price']
    #DF['trade_amt'] = DF['amountThisUpdate']
    #DF['90s_ret'] = (DF['adjMidF90s'] - DF['trade_price']) / DF['trade_price']   
    df = DF.reset_index(drop = True)
    df['interval'] = orange_interval(df['localTime'])
    df['trade_price'] = df['trade_amt_taking']/df['trade_qty_taking']
    df['order_amt'] = df['order_price'] * df['order_qty']

    mta = read_mta(date)
    
    df = df.merge(mta, on = ['interval', 'skey'], how = 'left')
    df['alpha'] = np.nan
    df['alpha'] = np.where(df['order_side'] == 1, df['alp_1d'] + 0.5*df['prev_yHatBuy90'], df['alpha'])
    df['alpha'] = np.where(df['order_side'] == 2, -df['alp_1d'] + 0.5*df['prev_yHatSell90'], df['alpha'])
    
    df['1d_ret'] = np.nan
    df['1d_ret'] = np.where(df['order_side'] == 1, (np.log(df['stk_dv_ND1_adj']/df['trade_price']) - df['beta'] * np.log(df['idx_dv_ND1']/df['ic'])), df['1d_ret'])
    df['1d_ret'] = np.where(df['order_side'] == 2, (np.log(df['trade_price']/df['stk_dv_ND1_adj']) - df['beta'] * np.log(df['ic']/df['idx_dv_ND1'])), df['1d_ret'])
    print(df)
    #df[['skey', 'date', 'ApplSeqNum', 'time', 'interval','clockAtArrival', 'cum_amount', 'bid1p', 'ask1p','bid1q', 'ask1q', 'prev_bid1p', 'prev_ask1p', 'prev_bid1q', 'prev_ask1q', 'order_side', 'order_price','order_amt', 'trade_price',	'order_qty', 'trade_qty_taking','trade_qty_making','trade_qty_cancel', 'prev_yHatBuy90', 'prev_yHatSell90','alp_1d','alpha', 'adjMidF90s', 'adjMidF300s','1d_ret', 'ic', 'idx_dv_ND1', 'stk_dv_ND1', 'stk_dv_ND1_adj', 'beta']].to_parquet('/data/home/jianw/OrderAnalysis/20211125_TakingBaseline/concat_data_all/%s.parquet'%(re.sub("[^0-9]", "", date)), index = False)
    #print(df.columns)
    #print(date)
    #print(df.loc[df['skey'] == 2300750])



before = datetime.datetime.now()          
if __name__ == '__main__':
    

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/monetization_research/rawData_jian/data_jian_20211213/'
    sdate = '2021-04-23'
    edate = '2021-04-23'

    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    '''
    P = mp.Pool(96)
    P.map(concat_stock, dates)
    P.close()
    P.join()
    '''
    concat_stock('2021-01-04')
    
    

after = datetime.datetime.now()
print(after - before)