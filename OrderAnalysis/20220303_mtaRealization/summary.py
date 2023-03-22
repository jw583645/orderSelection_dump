import numpy as np
import pandas as pd


Idxs = ['IF', 'IC', 'CSI1000', 'CSIRest']
Tops = [0.2]
Specs = ['top50sRet','bottom50sRet', 'middle50sRet', 'ALL']

def getSummary(spec):
    df_l = []
    for idx in Idxs:
        for top in Tops:
            df = pd.read_csv("/data/home/jianw/OrderAnalysis/20220303_mtaRealization/data/summaryData/%s.top%s.%s.csv"%(idx, top, spec))
            df['idx'] = idx
            df['top'] = top
            df_l.append(df.iloc[-1:])
    
    df = pd.concat(df_l)
    df_summary = df.groupby(['top'])['mean_30s', 'mean_50s', 'mean_1d', 'mean_30s_1d', 'mean_50s_1d', 'mean_trade_1d2', 'mean_sta'].agg('mean').reset_index()
    df = df.append(df_summary)
    print(df)
    df.to_csv('/data/home/jianw/OrderAnalysis/20220303_mtaRealization/data//summary_%s.csv' %spec, index = False)


for spec in Specs:
    getSummary(spec)
    