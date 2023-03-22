import os

sta_wt = 0.5
method = 'minute'
demean = 'minute'
order_side = 1


os.mkdir('/data/home/jianw/OrderAnalysis/20211125_TakingBaseline/stats_summary/baseline_POV/sta_wts/top_orders/staWt_%s_%s.%s.%s/' %(sta_wt, method, demean, order_side))