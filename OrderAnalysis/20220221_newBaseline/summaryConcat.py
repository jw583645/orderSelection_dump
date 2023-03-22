import numpy as np
import pandas as pd

univ = 'IF'
order_sides = [1]
methods = ['minute']
#rets = ['raw','minute']
rets = ['minute']
#mta_wts_30m = [0.0,0.2,0.4,0.6, 0.8,1.0]
sta_wts = [0.0,0.2,0.4,0.5, 0.6,0.8,1.0, 1.2, 1.4, 1.5,1.6, 1.8, 2.0]
mta_wts_30m = [0,0.2, 0.4,0.6, 0.8, 1]
#sta_wts = [0.0,0.5, 1.0, 1.5,2.0]
inSample = 'IS'
sdate = '2021-01-01'
edate = '2021-10-29'

df_list = []
for order_side in order_sides:
    for method in methods:
        for ret in rets:
            for mta_wt_30m in mta_wts_30m:
                for sta_wt in sta_wts:
                    df = pd.read_csv("/data/home/jianw/OrderAnalysis/20220221_newBaseline/stats_summary/%s/top.%s.%s.orderside%s.%s-%s.%s.%s.%s.csv"%(univ, method, ret, order_side, mta_wt_30m, sta_wt,sdate, edate, inSample))
                    df = df.loc[df['date']=='ALL']
                    df['selection'] = method
                    df['demean'] = ret
                    df['staWt'] = sta_wt
                    df['mta30mWt'] = mta_wt_30m
                    df_list.append(df)

df = pd.concat(df_list)
df['order_amt_all(yi)'] = df['order_amt_all']/1e8
df['trade_amt_all(yi)'] = df['trade_amt_all']/1e8
df['order_amt_top(yi)'] = df['order_amt_top']/1e8
df['trade_amt_top(yi)'] = df['trade_amt_top']/1e8
df['total_profit(Mn)'] = df['total_profit']/1e6

df['fillrate_all %'] = df['fillrate_all']*100
df['fillrate_top %'] = df['fillrate_top']*100

#df = df[['order_side', 'selection', 'demean', 'order_amt_all', 'trade_amt_all', 'fillrate_all', 'order_amt_top', 'trade_amt_top', 'fillrate_top','total_profit', 'return_bps', 'SR']]
df = df[['order_side', 'selection', 'demean', 'staWt', 'mta30mWt', 'order_amt_all(yi)', 'trade_amt_all(yi)', 'fillrate_all %', 'order_amt_top(yi)', 'trade_amt_top(yi)', 'fillrate_top %','total_profit(Mn)', 'return_bps', 'SR']]
df = round(df,2)
print(df)

df.to_csv('/data/home/jianw/OrderAnalysis/20220221_newBaseline/stats_summary/%s.top.baseline_summary_mta30mWts-staWts.orderside%s.%s.%s.%s.csv'%(univ, order_sides[0],sdate, edate, inSample), index = False)