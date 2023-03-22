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
import random
os.environ["OMP_NUM_THREADS"] = '1'


def concat_stock(tuples):
    IS, date = tuples
    print(date)
    Sample_dic = {1:'IS', 0: 'ALL', 2:'OOS'}
    try:
        os.remove('%s/%s/%s_%s.csv'%(data_path, re.sub("[^0-9]", "", date), date, Sample_dic[IS]))
        #os.remove('%s/%s/%s_%s.csv.gz'%(data_path, re.sub("[^0-9]", "", date), date, Sample_dic[IS]))
    except:
        pass
    skeys = os.listdir(data_path + re.sub("[^0-9]", "", date) +'/' )
    skeys = set(np.array([i[:7] for i in skeys]).astype(int))
    skeys = list(skeys)
    #print(skeys)
    skeys_IS = random.sample(skeys, int(len(skeys)*2/3))
    skeys_IS.sort()
    df = pd.DataFrame({'stocksIS': skeys_IS})
    print(df)
    df.to_csv('/data/home/jianw/OrderAnalysis/sampleStocks/stocksIS_%s.csv' %date, index = False)
    #return(df)

before = datetime.datetime.now()          
if __name__ == '__main__':

    RCH_DIR = '/data/rch/'
    data_path = '/data/home/jianw/OrderAnalysis/data/'
    sdate = '2021-01-01'
    edate = '2021-03-31'
    
    skeys_IS = set([2002048, 2000021, 2002185, 2002368, 2300316, 2002242, 2002002,
       2002465, 2002075, 2000997, 2002563, 2002174, 2002396, 2000553,
       2000990, 2002013, 2002302, 2000623, 2000009, 2002195, 2000709,
       2000717, 2000528, 2002506, 2002807, 2000564, 2002500, 2000830,
       2000778, 2000825, 2002745, 2000401, 2300296, 2300496, 2300223,
       2002946, 2000898, 2000600, 2000878, 2002085, 2000686, 2002074,
       2002399, 2002985, 2002110, 2002217, 2000967, 2002640, 2300376,
       2002791, 2000975, 2300459, 2300274, 2002273, 2002268, 2002390,
       2002941, 2000629, 2002603, 2300595, 2300134, 2000998, 2000158,
       2002867, 2000563, 2002948, 2002511, 2000739, 2000090, 2000598,
       2000031, 2000012, 2000877, 2002317, 2002233, 2002249, 2002353,
       2002709, 2000636, 2002373, 2002081, 2300024, 2002038, 2300418,
       2000999, 2000156, 2300001, 2000039, 2002244, 2000630, 2000738,
       2000028, 2300133, 2002797, 2002183, 2300026, 2002557, 2002010,
       2300166, 2002124, 2002966, 2000988, 2002416, 2000970, 2002957,
       2300463, 2002030, 2300271, 2002936, 2002705, 2000937, 2300699,
       2300009, 2002419, 2000027, 2002635, 2002156, 2002004, 2000540,
       2300012, 2300212, 2001872, 2002815, 2000813, 2000718, 2300113,
       2002064, 2002595, 2000960, 2300482, 2300180, 2002191, 2300146,
       2000807, 2000547, 2002382, 2000869, 2000061, 2300168, 2002572,
       2300285, 2002505, 2002056, 2002131, 2002920, 2002424, 2000930,
       2300017, 2002701, 2000543, 2000301, 2000959, 2002212, 2300357,
       2002028, 2300002, 2000685, 2002653, 2000932, 2000581, 2000008,
       2000681, 2002503, 2002408, 2000078])

    
    IS = 1
    
    date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
    date_list = date_list['date']
    date_list = date_list[(date_list >= sdate) & (date_list <= edate)]
    dates = date_list.to_list()
             
    tuples = []
    for date in dates:
        tuples.append((IS, date))
    P = mp.Pool(96)
    DF_l = P.map(concat_stock, tuples)
    P.close()
    P.join()

after = datetime.datetime.now()
print(after - before)