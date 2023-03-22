import numpy as np
import pandas as pd




price_level = [0,5,10,20,50, 100, 10000]
price_level_t = ['<5', '5~10', '10~20', '20~50', '50~100', '>100']
df_l = []

for j in range(10):
    
    print('decile %s'%(j+1))
    df = pd.read_parquet("/data/home/jianw/OrderAnalysis/20220117_30min//decile_data/orderside2/decile%s.parquet" %(j+1))
    df = df.sort_values(by = 'mid_price')
    
    df['order_amt_all'] = df['trade_amt'].sum()
    df['order_amt_cumsum_mta'] = df['trade_amt'].cumsum()
    df['price_quantile'] = (np.ceil(df['order_amt_cumsum_mta'] / df['order_amt_all'] * 10)).astype(int)
    print(df)
    '''
    print(sum(df['safeBid1p'] == 0))
    print(sum(df['safeAsk1p'] == 0))
    print(sum(0.5 * (df['safeAsk1p'] - df['safeBid1p']) / df['mid_price']  != df['halfSpread']))
    print(sum(0.5 * (df['safeAsk1p'] - df['safeBid1p']) / df['mid_price']  != df['halfSpread']))
    print(sum((np.log(df['stk_dv_ND1_adj']/df['mid_price']) - df['beta'] * np.log(df['idx_dv_ND1']/df['ic']))  != df['1d_ret']))
    '''
    #print(df[['1d_ret', 'halfSpread', 'mid_price']].corr())
    #print((df[['1d_ret', 'alp_1d', 'sta', 'halfSpread' ]]*10000).describe())
    
    
    
    
    
    trade_wts = []
    rets = []
    for i in range(10):
        #print('Price %s to %s' %(price_level[i], price_level[i+1]))
        #df_local = df.loc[(df['mid_price'] > price_level[i]) & (df['mid_price'] <=price_level[i+1])]
        df_local = df.loc[(df['price_quantile'] == i + 1)]
        trade_wt = (df_local['trade_amt']).sum() / (df['trade_amt']).sum()
        ret = (df_local['30m_ret'] * df_local['trade_amt']).sum() / (df_local['trade_amt']).sum() * 10000
        
        trade_wts.append(trade_wt)
        rets.append(ret)
        
    df = pd.DataFrame({'mta_q': j+1, 'price_level': range(1,11), 'trade_wts': trade_wts, 'rets (bps)' : rets})
    df_l.append(df)
    print(df)
    #df.to_csv('/data/home/jianw/OrderAnalysis/20211221_stamta_Interaction/decile_data/orderside1/price_level/price_level_decile%s.csv' %(j+1),index = False)
df = pd.concat(df_l)
df.to_csv('/data/home/jianw/OrderAnalysis/20220117_30min/decile_data/orderside1//price_level/price_level.csv' ,index = False)
#print(df)
#df.loc[df['date'] == 20210423].to_csv('/data/home/jianw/OrderAnalysis/20211221_stamta_Interaction/decile_data/orderside1/sample_0421.csv',index = False)