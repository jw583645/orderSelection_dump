import os
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
import multiprocessing as mp


def collapse_sta(skey, date):

    sta_path = '/data/home/jianw/sta/%s/%s.parquet' %(date, skey)
    df = pd.read_parquet(sta_path)
    assert df['ordering'].nunique() == df.shape[0]
    df['amountThisUpdate'] = df['cum_amount'] - df['prev_cum_amount']
    df['volumeThisUpdate'] = df['cum_volume'] - df['prev_cum_volume']
    assert (df['volumeThisUpdate'] == df['trade_qty']).all()
    df_multi = df[df.duplicated(['time'], keep = False)]
    df_single = df.loc[~df['ordering'].isin(df_multi['ordering'])]  # distinguish overlap orders
    d_sum = dict.fromkeys(['order_qty', 'trade_qty', 'amountThisUpdate'], 'sum')
    d_last = dict.fromkeys(['bid1p', 'ask1p', 'bid1q', 'ask1q', 'yHatBuy90', 'yHatSell90'], 'last') 
    d_first = dict.fromkeys(['prev_bid1p', 'prev_ask1p', 'prev_bid1q', 'prev_ask1q', 'prev_yHatBuy90', 'prev_yHatSell90', 'prev_time'], 'first')
    df_multi = df_multi.copy()
    df_multi[list(d_last.keys())] = df_multi.groupby('time')[list(d_last.keys())].transform('last')
    df_multi[list(d_first.keys())] = df_multi.groupby('time')[list(d_first.keys())].transform('first')
    d = dict.fromkeys(df_multi.columns , 'first')
    d.pop('time', None)
    d.pop('order_side', None)
    d.update(d_sum)
    d.update(d_last)   
    #print(df_multi)     
    df_multi = df_multi.groupby(['time', 'order_side'], as_index = False).agg(d) # merge overlap orders (with different order side)
    df = pd.concat([df_single, df_multi])
    df['trade_price'] = df['amountThisUpdate'] / df['trade_qty']
    return(df.sort_values('ordering').reset_index(drop = True))

    
def date_add_bar(date):
    date = str(date)
    return date[:4] + '-' + date[4:6] + '-' + date[6:]


def merge_alpha(pair):
    tstock, date = pair
    sta = collapse_sta(tstock, date)
    sta['prev_adjMidBuy'] = (sta['prev_yHatBuy90'] + 1) * sta['prev_ask1p']
    sta['prev_adjMidSell'] = sta['prev_bid1p'] / (sta['prev_yHatSell90'] + 1)
    sta['prev_yHat'] = (sta['prev_adjMidBuy'] + sta['prev_adjMidSell']) / 2
    sta = sta[(sta['prev_time'] > 93100000000)]
    sta.dropna(subset = ['prev_yHatBuy90', 'prev_yHatSell90'], inplace = True)
    sta = sta[(sta['prev_bid1p'] != 0) & (sta['prev_ask1p'] != 0)]
    mta = pd.read_parquet('/data/home/jianw/mta_l4po/data/rch/eq/mret_fwd/CHNUniv.EqALL4/1MIN/mret_fwd.CHNUniv.EqALL4.1MIN.{}.parquet'.format(date_add_bar(date)))
    stock_id = 'SZ' + str(tstock)[1:]
    mta = mta[mta['skey'] == stock_id].copy()

    if mta['beta'].isnull().any():
        print(pair, 'no beta')
        return

    if mta['stk_openv_ND1'].isnull().all():
        print(pair, 'no 1d return')
        return

    if mta['stk_openv_ND2'].isnull().all():
        print(pair, 'no 2d return')
        return

    mta['skey'] = int(tstock)
    mta['prev_time'] = mta['interval'].str[:2].astype(int) * 1e10 + mta['interval'].str[3:5].astype(int) * 1e8 + mta['interval'].str[6:8].astype(int) * 1e6
    sta['label'] = 0
    mta['label'] = 1
    mta['bid1p'] = np.where(mta['bid1p'].isnull(), mta['mid'], mta['bid1p']) 
    mta['ask1p'] = np.where(mta['ask1p'].isnull(), mta['mid'], mta['ask1p']) 
    mta = mta[mta.interval < '14:57:00']
    
    mta.rename(columns = {'mid': 'mta_mid_price', 'bid1p': 'mta_bid1p', 'ask1p': 'mta_ask1p'}, inplace = True)
    df = pd.concat([sta, mta[['skey', 'prev_time', 'min', 'alp_1d', 'alp_2d', 'beta', 'idx_openv_ND1', 'stk_openv_ND1', 
                              'idx_N30min', 'stk_N30min', 'stk_dclose_ND1', 'idx_dclose_ND1', 'idx_openv_ND2', 'stk_openv_ND2', 'mta_mid_price', 'mta_bid1p', 'mta_ask1p', 'label']]])
    assert df['skey'].nunique() == 1
    df.sort_values(['prev_time', 'label', 'ordering'], inplace = True)

    for col in ['min', 'alp_1d', 'alp_2d', 'beta', 'idx_openv_ND1', 'stk_openv_ND1', 'idx_N30min', 'stk_N30min', 'stk_dclose_ND1', 'idx_dclose_ND1', 'idx_openv_ND2', 'stk_openv_ND2', 'mta_mid_price', 'mta_bid1p', 'mta_ask1p']:
        df[col] = df[col].fillna(method = "ffill")

    df = df[df['label'] == 0]
    assert df.shape[0] == sta.shape[0]
    df = df.reset_index(drop = True)

    try:
        assert ~df[['prev_yHatBuy90', 'prev_yHatSell90', 'trade_price', 'prev_bid1p', 'prev_ask1p', 'prev_bid1q', 'prev_ask1q', 'prev_time', 'min', 'alp_1d', 'alp_2d', 'beta', 
    'idx_openv_ND1', 'stk_openv_ND1', 'idx_N30min', 'stk_N30min', 'stk_dclose_ND1', 'idx_dclose_ND1', 'idx_openv_ND2', 'stk_openv_ND2', 'mta_mid_price', 'mta_bid1p', 'mta_ask1p']].isnull().any().any()
    except:
        #print(df[['time', 'prev_time', 'prev_yHatBuy90', 'prev_yHatSell90', 'prev_bid1p', 'prev_ask1p']])
        print(pair)
        print(df[['prev_yHatBuy90', 'prev_yHatSell90', 'trade_price', 'prev_bid1p', 'prev_ask1p', 'prev_bid1q', 'prev_ask1q', 'prev_time', 'min', 'alp_1d', 'alp_2d', 'beta', 
        'idx_openv_ND1', 'stk_openv_ND1', 'idx_N30min', 'stk_N30min', 'stk_dclose_ND1', 'idx_dclose_ND1', 'idx_openv_ND2', 'stk_openv_ND2', 'mta_mid_price', 'mta_bid1p', 'mta_ask1p']].isnull().any())

    df['1d_ret'] = np.nan
    df['1d_ret'] = np.where(df['order_side'] == 1, (df['stk_openv_ND1'] / df['trade_price'] - 1) - df['beta'] * (df['idx_openv_ND1'] / df['ic'] - 1), df['1d_ret'])
    df['1d_ret'] = np.where(df['order_side'] == 2, (df['trade_price'] / df['stk_openv_ND1'] - 1) - df['beta'] * (df['ic'] / df['idx_openv_ND1'] - 1), df['1d_ret'])
    assert ~df['1d_ret'].isnull().any()
    df['2d_ret'] = np.nan
    df['2d_ret'] = np.where(df['order_side'] == 1, (df['stk_openv_ND2'] / df['trade_price'] - 1) - df['beta'] * (df['idx_openv_ND2'] / df['ic'] - 1), df['2d_ret'])
    df['2d_ret'] = np.where(df['order_side'] == 2, (df['trade_price'] / df['stk_openv_ND2'] - 1) - df['beta'] * (df['ic'] / df['idx_openv_ND2'] - 1), df['2d_ret'])
    assert ~df['2d_ret'].isnull().any()
    
    try:
        assert (df['mta_bid1p'] <= df['mta_mid_price'] + np.finfo(float).eps).all() and (df['mta_mid_price'] <= df['mta_ask1p'] + np.finfo(float).eps).all()
    except:
        print('bid ask issue', tstock, date)

    try:
        os.makedirs('/data/home/jianw/OrderAnalysis/data/{}'.format(date))
    except:
        pass

    df.to_csv('/data/home/jianw/OrderAnalysis/data/{}/{}.csv.gz'.format(date, tstock), index = False)


if __name__ == '__main__':

    sta_path = '/data/home/jianw/sta/'
    dates = os.listdir(sta_path)
    pairs = []

    for date in dates:
        stocks = os.listdir(sta_path + date)
        stocks = [x.split('.')[0] for x in stocks]

        for stock in stocks:
            pairs.append((stock, date))

    P = mp.Pool(96)
    P.map(merge_alpha, pairs)
    P.close()
    P.join()

    
    
    #merge_alpha((2000826, 20210107))






     