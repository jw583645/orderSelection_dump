import numpy as np
import pandas as pd
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import os, sys
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
import lightgbm as lgbm
os.environ["OMP_NUM_THREADS"] = '1'

def concat(date):
    print(date)
    df = pd.read_csv('%s/%s/%s_IS.csv'%(data_path, re.sub("[^0-9]", "", date), date) )
    df['trade_amt'] = np.minimum(df['trade_amt'], 1000000)
    df['netPnL'] = df['trade_amt'] * (df['1d_ret']- 0.0013)
    df = df[['skey','prev_yHatBuy90', 'alp_1d', 'trade_amt','1d_ret', 'netPnL','1d_jump_up_cur', '1d_jump_down_cur', '1d_jump_up_5min', '1d_jump_down_5min', 'prev_mid_up_cur',	'prev_mid_down_cur',	'prev_mid_up_10_orders',	'prev_mid_down_10_orders',	'sta_jump_up_cur',	'sta_jump_down_cur',	'sta_jump_up_10_orders',	'sta_jump_down_10_orders']]
    #df = df.groupby(['alp_1d', 'prev_yHatBuy90'], as_index = False).agg({'trade_amt':'sum', 'netPnL':'sum'})
    return(df)
def concat_oos(date):
    #print(date)
    df = pd.read_csv('%s/%s/%s_OOS.csv'%(data_path, re.sub("[^0-9]", "", date), date) )
    df['trade_amt'] = np.minimum(df['trade_amt'], 1000000)
    df['netPnL'] = df['trade_amt'] * (df['1d_ret']- 0.0013)
    df = df[['skey', 'prev_yHatBuy90', 'alp_1d', 'trade_amt','1d_ret', 'netPnL', '1d_jump_up_cur', '1d_jump_down_cur', '1d_jump_up_5min', '1d_jump_down_5min', 'prev_mid_up_cur',	'prev_mid_down_cur',	'prev_mid_up_10_orders',	'prev_mid_down_10_orders',	'sta_jump_up_cur',	'sta_jump_down_cur',	'sta_jump_up_10_orders',	'sta_jump_down_10_orders']]
    #df = df.groupby(['alp_1d', 'prev_yHatBuy90'], as_index = False).agg({'trade_amt':'sum', 'netPnL':'sum'})
    return(df)

    
if __name__ == '__main__':

    data_path = '/data/home/jianw/OrderAnalysis/data_feature/'
    RCH_DIR= '/data/rch/'
    sdate = '2021-01-01'
    edate = '2021-03-31'
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    P = mp.Pool(96)
    DF_l = P.map(concat, dates)
    P.close()
    P.join()
    df = pd.concat(DF_l)
    print(df)
    df['ret_1d'] = df['1d_ret']
    df['alp_1d_sq'] = df['alp_1d']**2
    df['prev_yHatBuy90_sq'] = df['prev_yHatBuy90']**2
    df['alp_1d_prev_yHatBuy90'] = df['alp_1d'] * df['prev_yHatBuy90']
    df_mta = pd.read_csv("/data/home/jianw/OrderAnalysis/mta_quality/df.csv")
    df = df.merge(df_mta, on = 'skey', how = 'left')
    df['corr'] = df['corr'].fillna(0)
    print(df)
    #df['open_10m'] = (df['min']<=10).astype(int)
    #df['last_10m'] = (df['min']>=227).astype(int)
    

    X = df[['alp_1d', 'prev_yHatBuy90' , 'alp_1d_prev_yHatBuy90' , 'trade_amt',  '1d_jump_up_5min', '1d_jump_down_5min', 	'prev_mid_up_10_orders',	'prev_mid_down_10_orders',	'sta_jump_up_10_orders',	'sta_jump_down_10_orders'
]]
    #X = df[['alp_1d', 'prev_yHatBuy90' , 'trade_amt' ]]
    Y = df['ret_1d']
    X = sm.add_constant(X)
    wls_model = sm.WLS(Y,X, weights=df['trade_amt'] ** 0.5)
    #wls_model = sm.WLS(Y,X)
    result = wls_model.fit()
    
    #result = sm.WLS(formula="ret_1d ~ alp_1d + prev_yHatBuy90 + alp_1d_sq + prev_yHatBuy90_sq + alp_1d_prev_yHatBuy90 + trade_amt", weights = df['trade_amt']  data=df).fit()
    print(result.params)
    print(result.summary())
    print('!!!!!!!!!!!!!!!!!!!!!!! IS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'netPnL'].sum()/58)
    print(df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'trade_amt'].sum()/58)
    print(df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'trade_amt'].sum()/df['trade_amt'].sum())
    trade_total = df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'trade_amt'].sum()
    
    df['yhat'] = result.predict()
    df = df.sort_values('yhat', ascending = False)
    df['trade_amt_cum'] = df['trade_amt'].cumsum()
    print(df.loc[df['trade_amt_cum'] <= trade_total]['netPnL'].sum()/58)
    
    
    print('!!!!!!!!!!!!!!!!!!!!!!! OOS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    data_path = '/data/home/jianw/OrderAnalysis/data_feature/'
    RCH_DIR= '/data/rch/'
    sdate = '2021-01-01'
    edate = '2021-03-31'
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    P = mp.Pool(96)
    DF_l = P.map(concat_oos, dates)
    P.close()
    P.join()
    df = pd.concat(DF_l)
    print(df)
    df['ret_1d'] = df['1d_ret']
    df['alp_1d_sq'] = df['alp_1d']**2
    df['prev_yHatBuy90_sq'] = df['prev_yHatBuy90']**2
    df['alp_1d_prev_yHatBuy90'] = df['alp_1d'] * df['prev_yHatBuy90']
    df = df.merge(df_mta, on = 'skey', how = 'left')
    df['corr'] = df['corr'].fillna(0)
    
    X = df[['alp_1d', 'prev_yHatBuy90' , 'alp_1d_prev_yHatBuy90' , 'trade_amt',  '1d_jump_up_5min', '1d_jump_down_5min', 	'prev_mid_up_10_orders',	'prev_mid_down_10_orders',	'sta_jump_up_10_orders',	'sta_jump_down_10_orders'
]]
    X = sm.add_constant(X)
    
    
    print(df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'netPnL'].sum()/58)
    print(df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'trade_amt'].sum()/58)
    trade_total = df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'trade_amt'].sum()
    
    df['yhat'] = result.predict(X)
    df = df.sort_values('yhat', ascending = False)
    df['trade_amt_cum'] = df['trade_amt'].cumsum()
    print(df.loc[df['trade_amt_cum'] <= trade_total]['netPnL'].sum()/58)
    