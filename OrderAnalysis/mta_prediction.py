import os
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
import multiprocessing as mp

def date_add_bar(date):
    date = str(date)
    return date[:4] + '-' + date[4:6] + '-' + date[6:]


def mta_quality(date):
    df = pd.read_parquet('/data/home/jianw/mta_l4po/data/rch/eq/mret_fwd/CHNUniv.EqALL4/1MIN/mret_fwd.CHNUniv.EqALL4.1MIN.{}.parquet'.format(date))
    df = df[df.interval < '14:57:00']
    for col in ['min', 'alp_1d', 'alp_2d', 'beta', 'idx_openv_ND1', 'stk_openv_ND1', 'idx_N30min', 'stk_N30min', 'stk_dclose_ND1', 'idx_dclose_ND1', 'idx_openv_ND2', 'stk_openv_ND2']:
        df[col] = df[col].fillna(method = "ffill")
    #print(df.loc[df['interval'] == '09:31:00'][['skey', 'date', 'interval','idx','idx_openv_ND1']])
    df['1d_ret'] = (df['stk_openv_ND1'] / df['mid'] - 1) - df['beta'] * (df['idx_openv_ND1'] / df['idx'] - 1)
    #df['1d_ret'] = np.where(df['order_side'] == 2, (df['trade_price'] / df['stk_openv_ND1'] - 1) - df['beta'] * (df['ic'] / df['idx_openv_ND1'] - 1), df['1d_ret'])

    return(df[['skey', 'interval', 'date', 'alp_1d', '1d_ret']])

def mta_corr(skey):
    print(skey[0:8])
    df = pd.read_csv('/data/home/jianw/OrderAnalysis/mta_quality/%s' %skey)
    cor = df[['alp_1d','1d_ret']].corr().iloc[0,1]
    return(pd.DataFrame({'skey': skey[0:8], 'corr': cor}, index = [0]))


if __name__ == '__main__':

    sta_path = '/data/home/jianw/sta/'
    dates = os.listdir(sta_path)
    pairs = []
    #mta_quality('2021-03-31')
    RCH_DIR= '/data/rch/'
    sdate = '2020-07-01'
    edate = '2020-12-31'
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    skeys = os.listdir("/data/home/jianw/OrderAnalysis/mta_quality/")
    
    P = mp.Pool(96)
    DF_l = P.map(mta_corr, skeys)
    P.close()
    P.join()
    df = pd.concat(DF_l)
    print(df)
    df['skey'] = np.where(df['skey'].str[0:2]=='SH', '1' + df['skey'].str[2:9], '2' + df['skey'].str[2:9]).astype(int)
    df.to_csv('/data/home/jianw/OrderAnalysis/mta_quality/df.csv', index = False)
    
    
    '''
    P = mp.Pool(96)
    DF_l = P.map(mta_quality, dates)
    P.close()
    P.join()
    df = pd.concat(DF_l)
    print(df)
    
    d = dict(tuple(df.groupby('skey')))
    for key in d.keys():
        print(d[key])
        d[key].to_csv('/data/home/jianw/OrderAnalysis/mta_quality/%s.csv' %key, index = False)
    '''
    '''
    print(df[['alp_1d', '1d_ret']].corr())
    Corr = []
    for skey in set(df['skey']):
        print(skey)
        df_s = df.loc[df['skey']==skey]
        cor = df_s[['alp_1d', '1d_ret']].iloc[0,1]
        Corr.append(cor)
        print(cor)
    print(Corr)
    '''

