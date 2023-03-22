import os
import re
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
import multiprocessing as mp


threshold = 0.1
date = 20210104
def date_add_bar(date):
    date = str(date)
    return date[:4] + '-' + date[4:6] + '-' + date[6:]
    
def mta_filter(date):  
    print(date) 
    ic = pd.read_csv("/data/rch/raw/secData_TR/index_weight/weight_table_IC_%s.csv.gz"%format(date_add_bar(date)))
    mta = pd.read_parquet('/data/home/jianw/mta_l4po/data/rch/eq/mret_fwd/CHNUniv.EqALL4/1MIN/mret_fwd.CHNUniv.EqALL4.1MIN.{}.parquet'.format(date_add_bar(date)))
    mta = mta.loc[mta['skey'].isin(ic.ID)][['skey', 'date', 'interval', 'alp_1d']]
    
    mta['top_alp'] = mta.groupby('interval', as_index = False).alp_1d.transform(lambda x: np.quantile(x, 1- threshold)) 
    mta['buttom_alp'] = mta.groupby('interval', as_index = False).alp_1d.transform(lambda x: np.quantile(x, threshold)) 
    
    mta['top'] = (mta['alp_1d'] >= mta['top_alp']).astype(int)
    mta['buttom'] = (mta['alp_1d'] <= mta['buttom_alp']).astype(int)
    mta['interval'] = mta['interval'].str.replace(':', '').astype(int)*1e6
    mta['skey'] = np.where(mta['skey'].str[:2] == 'SH',  '1' + mta['skey'].str[2:], '2' + mta['skey'].str[2:])
    mta.to_csv('/data/home/jianw/sta/ICTopNames/ICTopButtom_%s.csv'%date, index = False)
    print(mta)
#print(mta)

if __name__ == '__main__':

    sta_path = '/data/home/jianw/sta/data/'
    dates = os.listdir(sta_path)
    print(dates)

    
    P = mp.Pool(96)
    P.map(mta_filter, dates)
    P.close()
    P.join()
    