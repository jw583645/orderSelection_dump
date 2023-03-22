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
#import datetime
#from datetime import datetime
os.environ["OMP_NUM_THREADS"] = '1'


def fromtimestamp(timestamp): 
    return (datetime.fromtimestamp(timestamp/1e6).time())
    try:
        return (datetime.fromtimestamp(timestamp/1e6).time())
    except:
        return np.nan

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
    # read 30m mta
    mta_30m = pd.read_csv('/data/home/jianw/mta_eq_cn/data/rch/eq/sig/CHNUniv.EqALL4/1MIN/CHNOR001/1.3.1c/alp.hzn30m/alp.CHNOR001.%s.csv.gz' %date)
    mta_30m = mta_30m.rename(columns = {'yhat.mret_fT0_30m': 'alp_30m'})
    mta = mta.merge(mta_30m, on = ['skey', 'date' ,'interval'], how = 'left')

    print(date, len(list(set(mta['skey']))))
    mta['interval'] = (mta['interval'].str.replace(':', '')).astype(int)*1e6
    mta['skey'] = np.where(mta['skey'].str.slice(0,2) == 'SH', '1' + mta['skey'].str.slice(2), '2' + mta['skey'].str.slice(2)).astype(int)
    return mta[['skey', 'interval', 'alp_1d', 'alp_2d', 'alp_30m', 'beta', 'idx_dv_ND1', 'stk_dv_ND1', 'idx_N30min', 'stk_N30min', 'stk_dclose_ND1', 'idx_dv_ND2', 'stk_dv_ND2', 'idx_dclose_ND1', 'stk_dclose_ND1_adj', 'stk_dv_ND1_adj']]
     
    
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
    ## calculate some statistics
    df['localTime'] = (df['localClockAtArrival']).apply(fromtimestamp).astype(str).str.replace(":","").astype(float)*1e6
    df['interval'] = orange_interval(df['localTime'])
    df['trade_price'] = df['trade_amt_taking']/df['trade_qty_taking']
    df['order_amt'] = df['order_price'] * df['order_qty']
    df['index_forward30m_ret'] = np.log(df['ic_30m']/df['ic'])
    

    mta = read_mta(date)
    
    df = df.merge(mta, on = ['interval', 'skey'], how = 'left')
    #df['alpha'] = np.nan
    #df['alpha'] = np.where(df['order_side'] == 1, df['alp_1d'] + 0.5*df['prev_yHatBuy90'], df['alpha'])
    #df['alpha'] = np.where(df['order_side'] == 2, -df['alp_1d'] + 0.5*df['prev_yHatSell90'], df['alpha'])
    # 1d ret
    df['1d_ret'] = np.nan
    df['1d_ret'] = np.where(df['order_side'] == 1, (np.log(df['stk_dv_ND1_adj']/df['trade_price']) - df['beta'] * np.log(df['idx_dv_ND1']/df['ic'])), df['1d_ret'])
    df['1d_ret'] = np.where(df['order_side'] == 2, (np.log(df['trade_price']/df['stk_dv_ND1_adj']) - df['beta'] * np.log(df['ic']/df['idx_dv_ND1'])), df['1d_ret'])
    
    # 30m ret
    #df['safeBid1p_30m'] = np.where(df.eval('bid1p_30m==0'), df['ask1p_30m'], df['bid1p_30m'])
    #df['safeAsk1p_30m'] = np.where(df.eval('ask1p_30m==0'), df['bid1p_30m'], df['ask1p_30m'])
    #df['mid_price_30m'] = df.eval('(safeBid1p_30m * ask1q_30m + safeAsk1p_30m * bid1q_30m) / (bid1q_30m + ask1q_30m)')
   
    df['30m_ret'] = np.nan
    df['30m_ret'] = np.where(df['order_side'] == 1, (np.log(df['adjMid_30m']/df['trade_price']) - df['beta'] * df['index_forward30m_ret']), df['30m_ret'])
    df['30m_ret'] = np.where(df['order_side'] == 2, (np.log(df['trade_price']/df['adjMid_30m']) + df['beta'] * df['index_forward30m_ret']), df['30m_ret'])
    
    print(df)
    df[['skey', 'date', 'ApplSeqNum', 'time', 'interval','clockAtArrival', 'cum_amount', 'bid1p', 'ask1p','bid1q', 'ask1q', 'prev_bid1p', 'prev_ask1p', 'prev_bid1q', 'prev_ask1q', 'order_side', 'order_price','order_amt', 'trade_price',	'order_qty', 'trade_qty_taking','trade_qty_making','trade_qty_cancel', 'prev_yHatBuy90', 'prev_yHatSell90','alp_1d', 'alp_30m',  'adjMidF90s', 'adjMidF300s', 'adjMid_30m', '1d_ret', '30m_ret', 'ic','ic_30m','ic_-5s', 'ic_-30s', 'index_forward30m_ret', 'adjMid_-5s', 'bid1p_-5s', 'ask1p_-5s', 'adjMid_-30s', 'bid1p_-30s', 'ask1p_-30s',  'idx_dv_ND1', 'stk_dv_ND1', 'stk_dv_ND1_adj', 'beta']].to_parquet('/data/home/jianw/monetization_research/rawData_jian/concat_data/IC/%s.parquet'%(re.sub("[^0-9]", "", date)), index = False)
    #print(df.columns)
    #print(date)
    #print(df.loc[df['skey'] == 2300750])



before = datetime.now()          
if __name__ == '__main__':
    

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/monetization_research/rawData_jian/data_jian_20220119/'
    sdate = '2021-01-04'
    edate = '2021-10-29'

    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    #concat_stock('2021-01-04')
    
    P = mp.Pool(96)
    P.map(concat_stock, dates)
    P.close()
    P.join()
    
    
    #read_mta('2021-01-25')
    
    

after = datetime.now()
print(after - before)