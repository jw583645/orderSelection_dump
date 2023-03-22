#!/usr/bin/env python
# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
import multiprocessing as mp


files = os.listdir('/data/home/qingyun/project_mkt_impact/insample_data/')
dates = np.array(sorted([x.split('.')[0] for x in files]))


def benchmark_day(date):
    
    df = pd.read_parquet('/data/home/qingyun/project_mkt_impact/insample_data/{}.parquet'.format(date))
    df = df[df.order_side == 1]
    pnl_table = {}
    
    for k in range(11):
        df['alpha_k'] = df['alp_1d'] + 0.1 * k * df['prev_yHatBuy90']
        df_trade = df[df['alpha_k'] > 10 * 1e-4].copy()
        #df_trade['pnl'] = df_trade['amountThisUpdate'] * (df_trade['1d_ret'] - 13 * 1e-4)
        df_trade['pnl'] = df_trade['trade_qty'] * df_trade['trade_price'] * (df_trade['1d_ret'] - 13 * 1e-4)
        pnl_table[k] = df_trade['pnl'].sum()
        
    pnl_df = pd.DataFrame(pnl_table, index = [date])
    return pnl_df


if __name__ == '__main__':
    
    P = mp.Pool(96)
    res = P.map(benchmark_day, dates)
    P.close()
    P.join()
    df = pd.concat(res)
    df.to_csv('/data/home/qingyun/project_mkt_impact/output/benchmark_pnl_10_j.csv.gz', index = False)
    
    #print(benchmark_day(20210114))
    





