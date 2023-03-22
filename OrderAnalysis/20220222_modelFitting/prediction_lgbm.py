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

def loadDayOrder(pair):
    print(pair)
    univ, date = pair
    print(date)
    df = pd.read_parquet('/data/home/jianw/monetization_research/rawData_jian/concat_data/%s/%s.parquet'%(univ,  date))
    return df

def loadOrders(univ, order_side, sdate, edate, period):
    if period == 'training':
        dates  = pd.read_csv("/data/home/jianw/OrderAnalysis/evendates.csv")
        #date_list = date_list['0'].astype(str)
        dates = dates.loc[(dates['0']>=sdate) & (dates['0']<=edate)]['0']
        pairs = []
        for date in dates:
            pairs.append((univ, date))
        #print(pairs)
        
        P = mp.Pool(96)
        DF_l = P.map(loadDayOrder, pairs)
        P.close()
        P.join()
        df = pd.concat(DF_l)
        
        return(df)


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
    sdate = 20210101
    edate = 20210131
    univ = 'IC'
    order_side = 1
    period = 'training'
    print(loadOrders(univ, order_side, sdate, edate, period))
    '''
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

    #df['open_10m'] = (df['min']<=10).astype(int)
    #df['last_10m'] = (df['min']>=227).astype(int)
    
    
    
    '''
    '''

    X_train = df[['alp_1d', 'prev_yHatBuy90' , 'alp_1d_prev_yHatBuy90' , 'trade_amt',  '1d_jump_up_5min', '1d_jump_down_5min', 	'prev_mid_up_10_orders',	'prev_mid_down_10_orders',	'sta_jump_up_10_orders',	'sta_jump_down_10_orders'
]]
    #X = df[['alp_1d', 'prev_yHatBuy90' , 'trade_amt' ]]
    Y_train = df['ret_1d']
    
    
    P = mp.Pool(96)
    DF_l = P.map(concat_oos, dates)
    P.close()
    P.join()
    df_oos = pd.concat(DF_l)
    df_oos['ret_1d'] = df_oos['1d_ret']
    df_oos['alp_1d_sq'] = df_oos['alp_1d']**2
    df_oos['prev_yHatBuy90_sq'] = df_oos['prev_yHatBuy90']**2
    df_oos['alp_1d_prev_yHatBuy90'] = df_oos['alp_1d'] * df_oos['prev_yHatBuy90']
    
    X_test = df_oos[['alp_1d', 'prev_yHatBuy90' , 'alp_1d_prev_yHatBuy90' , 'trade_amt',  '1d_jump_up_5min', '1d_jump_down_5min', 	'prev_mid_up_10_orders',	'prev_mid_down_10_orders',	'sta_jump_up_10_orders',	'sta_jump_down_10_orders'
]]
    Y_test = df_oos['ret_1d']
    
    print("Train/Test Sizes : ", X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    train_dataset = lgbm.Dataset(X_train, Y_train, feature_name=['alp_1d', 'prev_yHatBuy90' , 'alp_1d_prev_yHatBuy90' , 'trade_amt',  '1d_jump_up_5min', '1d_jump_down_5min', 	'prev_mid_up_10_orders',	'prev_mid_down_10_orders',	'sta_jump_up_10_orders',	'sta_jump_down_10_orders'
])
    test_dataset = lgbm.Dataset(X_test, Y_test, feature_name=['alp_1d', 'prev_yHatBuy90' , 'alp_1d_prev_yHatBuy90' , 'trade_amt',  '1d_jump_up_5min', '1d_jump_down_5min', 	'prev_mid_up_10_orders',	'prev_mid_down_10_orders',	'sta_jump_up_10_orders',	'sta_jump_down_10_orders'
])

    booster = lgbm.train({"objective": "regression"},
                    train_set=train_dataset, valid_sets=(test_dataset,),
                    num_boost_round=10)
    
    test_preds = booster.predict(X_test)
    train_preds = booster.predict(X_train)

    print("\nTest  R2 Score : %.2f"%r2_score(Y_test, test_preds))
    print("Train R2 Score : %.2f"%r2_score(Y_train, train_preds))


    print('!!!!!!!!!!!!!!!!!!!!!!! IS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'netPnL'].sum()/58)
    print(df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'trade_amt'].sum()/58)
    print(df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'trade_amt'].sum()/df['trade_amt'].sum())
    trade_total = df.loc[df['alp_1d'] + 0.5 * df['prev_yHatBuy90'] > 10/10000, 'trade_amt'].sum()
    
    df['yhat'] = train_preds
    df = df.sort_values('yhat', ascending = False)
    df['trade_amt_cum'] = df['trade_amt'].cumsum()
    print(df.loc[df['trade_amt_cum'] <= trade_total]['netPnL'].sum()/58)

    
    print('!!!!!!!!!!!!!!!!!!!!!!! OOS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(df_oos.loc[df_oos['alp_1d'] + 0.5 * df_oos['prev_yHatBuy90'] > 10/10000, 'netPnL'].sum()/58)
    print(df_oos.loc[df_oos['alp_1d'] + 0.5 * df_oos['prev_yHatBuy90'] > 10/10000, 'trade_amt'].sum()/58)
    trade_total = df_oos.loc[df_oos['alp_1d'] + 0.5 * df_oos['prev_yHatBuy90'] > 10/10000, 'trade_amt'].sum()
    
    df_oos['yhat'] = test_preds
    df_oos = df_oos.sort_values('yhat', ascending = False)
    df_oos['trade_amt_cum'] = df_oos['trade_amt'].cumsum()
    print(df_oos.loc[df_oos['trade_amt_cum'] <= trade_total]['netPnL'].sum()/58)
    '''
    