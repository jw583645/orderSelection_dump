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


def calcOverlap(date):
    print(date)
    try:
        df1 = pd.read_parquet('/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/selectedOrders/%s/baseline/topOrders.%s.parquet' %(univ, date))
    except:
        pass
        return None
    df1 = df1.groupby(['interval', 'skey'])['order_amt'].agg('sum').reset_index()
    
    df2 = pd.read_parquet('/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/selectedOrders/%s/realization/topOrders.%s.parquet' %(univ, date))
    df2 = df2.groupby(['interval', 'skey'])['order_amt'].agg('sum').reset_index()
    
    df = pd.merge(df1, df2, on = ['interval', 'skey'], how = 'left' )
    df = df.fillna(0)
    df['order_amt_overlap'] = np.minimum(df['order_amt_x'], df['order_amt_y'])
    

    return(pd.DataFrame({'date': date, 'order_amt_base': df['order_amt_x'].sum(), 'order_amt_overlap' : df['order_amt_overlap'].sum(), 'overlap': df['order_amt_overlap'].sum()/df['order_amt_x'].sum() }, index = [0]))
    
def calcPrice(date):
    print(date)
    try:
        df1 = pd.read_parquet('/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/selectedOrders/%s/baseline/topOrders.%s.parquet' %(univ, date))
        #df2 = pd.read_parquet('/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/selectedOrders/%s/1-1/topOrders.%s.parquet' %(univ, date))
        df3 = pd.read_parquet('/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/selectedOrders/%s/realization/topOrders.%s.parquet' %(univ, date))
    except:
        pass
        return None
    price1 = np.average(df1['prev_ask1p'], weights = df1['order_amt'])
    #price2 = np.average(df2['prev_ask1p'], weights = df2['order_amt'])
    price3 = np.average(df3['prev_ask1p'], weights = df3['order_amt'])
    

    #return(pd.DataFrame({'date': date, 'price1': price1, 'price2': price2, 'price3': price3, 'OrderAmt1':df1['order_amt'].sum(), 'OrderAmt2':df2['order_amt'].sum(),
    #'OrderAmt3':df3['order_amt'].sum() }, index = [0]))
    return(pd.DataFrame({'date': date, 'price1': price1, 'price3': price3, 'OrderAmt1':df1['order_amt'].sum(), 'OrderAmt3':df3['order_amt'].sum(),
     }, index = [0]))

def calcPriceWt(date):
    print(date)
    try:
        df = pd.read_parquet('/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/selectedOrders/%s/realization/topOrders.%s.parquet' %(univ, date))
    except:
        pass
        return None
    #orderAmt = df['order_amt'].sum()
    OrderAmts = []
    for i in range(len(price_level)-1):
        OrderAmts.append(df.loc[(df['prev_ask1p'] < price_level[i+1]) & (df['prev_ask1p'] >= price_level[i]) , 'order_amt'].sum())
        
        print(price_level[i], price_level[i+1])
    orderAmt = pd.DataFrame(OrderAmts).transpose()
    df = pd.DataFrame({'date': date, 'univ': univ}, index = [0] )
    df = pd.concat([df, orderAmt], axis=1)
    return(df)
    
    





before = datetime.datetime.now()          
if __name__ == '__main__':

    RCH_DIR = '/data/rch/'
    univ = 'IF'
    sdate = '2021-01-01'
    edate = '2021-10-29'
    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
    
    price_level = [0, 5, 10, 20, 50, 100, 10000]
    
    P = mp.Pool(96)
    df_list = P.map(calcPrice, dates)
    P.close()
    P.join()
    
    df = pd.concat(df_list)
    #df = df.append(df.mean(numeric_only=True), ignore_index=True)
    
    
    df_summary = pd.DataFrame({'price1':np.average(df['price1'], weights = df['OrderAmt1']), 'price3':np.average(df['price3'], weights = df['OrderAmt3']) },  index = [0])
    df = df.append(df_summary)
    
    '''
    print(df)
    df.to_csv('/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/selectedOrders/%s.pricePortion1.csv'%univ, index = False)
    '''
    df.to_csv('/data/home/jianw/OrderAnalysis/20220321_mtaRealization_improve/selectedOrders/%s.weightedPrice.csv'%univ, index = False)
    #calcPriceWt('2021-01-04')

after = datetime.datetime.now()
print(after - before)