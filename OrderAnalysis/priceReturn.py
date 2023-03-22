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

sdate = '2020-01-01'
edate = '2022-02-10'
RCH_DIR = '/data/rch/'
date_list = pd.read_csv("%s/raw/secData_TR/tradeDate.csv" %RCH_DIR, header=0)
date_list = date_list['date']
date_list = date_list[(date_list >= sdate) & (date_list <= edate)]

Indices = ['IF', 'IC' ,'CSI1000', 'CSIRest']

IF_cheap_amt = []
IF_cheap_ret = []
IF_exp_amt = []
IF_exp_ret = []

IC_cheap_amt = []
IC_cheap_ret = []
IC_exp_amt = []
IC_exp_ret = []

CSI1000_cheap_amt = []
CSI1000_cheap_ret = []
CSI1000_exp_amt = []
CSI1000_exp_ret = []

CSIRest_cheap_amt = []
CSIRest_cheap_ret = []
CSIRest_exp_amt = []
CSIRest_exp_ret = []

price_cut = 15

for date in date_list:
    print(date)
    df = pd.read_csv("/data/rch/raw/marketData_TR/dailyData/stock/%s/stock_%s.csv.gz"%(date[0:4], date))
    IFMember = pd.read_csv("/data/rch/raw/secData_TR/index_member/member_table_IF_%s.csv.gz"%date)['ID']
    ICMember = pd.read_csv("/data/rch/raw/secData_TR/index_member/member_table_IC_%s.csv.gz"%date)['ID']
    CSI1000Member = pd.read_csv("/data/rch/raw/secData_TR/index_member/member_table_CSI1000_%s.csv.gz"%date)['ID']
    CSIRestMember = pd.read_csv("/data/rch/raw/secData_TR/index_member/member_table_CSIRest_%s.csv.gz"%date)['ID']
    if_cheap = df.loc[(df['ID'].isin(IFMember)) & (df['yclose']<price_cut)]
    if_exp = df.loc[(df['ID'].isin(IFMember)) & (df['yclose']>=price_cut)]
    ic_cheap = df.loc[(df['ID'].isin(ICMember)) & (df['yclose']<price_cut)]
    ic_exp = df.loc[(df['ID'].isin(ICMember)) & (df['yclose']>=price_cut)]
    csi1000_cheap = df.loc[(df['ID'].isin(CSI1000Member)) & (df['yclose']<price_cut)]
    csi1000_exp = df.loc[(df['ID'].isin(CSI1000Member)) & (df['yclose']>=price_cut)]
    csirest_cheap = df.loc[(df['ID'].isin(CSIRestMember)) & (df['yclose']<price_cut)]
    csirest_exp = df.loc[(df['ID'].isin(CSIRestMember)) & (df['yclose']>=price_cut)]
    
    ## calculate return
    if_cheap_amt = if_cheap['amount'].sum()
    if_cheap_ret = (if_cheap['amount'] * if_cheap['dayReturn']).sum() / if_cheap_amt
    if_exp_amt = if_exp['amount'].sum()
    if_exp_ret = (if_exp['amount'] * if_exp['dayReturn']).sum() / if_exp_amt
    IF_cheap_amt.append(if_cheap_amt)
    IF_cheap_ret.append(if_cheap_ret)
    IF_exp_amt.append(if_exp_amt)
    IF_exp_ret.append(if_exp_ret)
    
    ic_cheap_amt = ic_cheap['amount'].sum()
    ic_cheap_ret = (ic_cheap['amount'] * ic_cheap['dayReturn']).sum() / ic_cheap_amt
    ic_exp_amt = ic_exp['amount'].sum()
    ic_exp_ret = (ic_exp['amount'] * ic_exp['dayReturn']).sum() / ic_exp_amt
    IC_cheap_amt.append(ic_cheap_amt)
    IC_cheap_ret.append(ic_cheap_ret)
    IC_exp_amt.append(ic_exp_amt)
    IC_exp_ret.append(ic_exp_ret)
    
    csi1000_cheap_amt = csi1000_cheap['amount'].sum()
    csi1000_cheap_ret = (csi1000_cheap['amount'] * csi1000_cheap['dayReturn']).sum() / csi1000_cheap_amt
    csi1000_exp_amt = csi1000_exp['amount'].sum()
    csi1000_exp_ret = (csi1000_exp['amount'] * csi1000_exp['dayReturn']).sum() / csi1000_exp_amt
    CSI1000_cheap_amt.append(csi1000_cheap_amt)
    CSI1000_cheap_ret.append(csi1000_cheap_ret)
    CSI1000_exp_amt.append(csi1000_exp_amt)
    CSI1000_exp_ret.append(csi1000_exp_ret)
    
    csirest_cheap_amt = csirest_cheap['amount'].sum()
    csirest_cheap_ret = (csirest_cheap['amount'] * csirest_cheap['dayReturn']).sum() / csirest_cheap_amt
    csirest_exp_amt = csirest_exp['amount'].sum()
    csirest_exp_ret = (csirest_exp['amount'] * csirest_exp['dayReturn']).sum() / csirest_exp_amt
    CSIRest_cheap_amt.append(csirest_cheap_amt)
    CSIRest_cheap_ret.append(csirest_cheap_ret)
    CSIRest_exp_amt.append(csirest_exp_amt)
    CSIRest_exp_ret.append(csirest_exp_ret)
    

DF_dic = {'date':date_list, 'IF_cheap_amt': IF_cheap_amt, 'IF_cheap_ret': IF_cheap_ret, 'IF_exp_amt': IF_exp_amt, 'IF_exp_ret': IF_exp_ret, 'IC_cheap_amt': IC_cheap_amt, 'IC_cheap_ret': IC_cheap_ret, 'IC_exp_amt': IC_exp_amt, 'IC_exp_ret': IC_exp_ret, 'CSI1000_cheap_amt': CSI1000_cheap_amt, 'CSI1000_cheap_ret': CSI1000_cheap_ret, 'CSI1000_exp_amt': CSI1000_exp_amt, 'CSI1000_exp_ret': CSI1000_exp_ret, 'CSIRest_cheap_amt': CSIRest_cheap_amt, 'CSIRest_cheap_ret': CSIRest_cheap_ret, 'CSIRest_exp_amt': CSIRest_exp_amt, 'CSIRest_exp_ret': CSIRest_exp_ret }

df = pd.DataFrame(DF_dic)
print(df)
df.to_csv('/data/home/jianw/OrderAnalysis/priceRet.pricecut%s.csv' %(price_cut),index = False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    