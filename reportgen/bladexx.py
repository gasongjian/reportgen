# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 22:00:26 2017

@author: gason
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import report as rpt
reload(rpt)



#  数据d导入
code=rpt.read_code('.\\data\\code.xlsx')
data0=pd.read_excel('.\\data\\data.xlsx',encoding='gbk')

# 数据清洗
data=data0[(data0['Q5']==1)|(data0['Q5'].isnull())]#清楚自己购买但使用不是自己的人
data=data[data[u'来源详情']==u'直接访问']
# 描述统计
filename=u'小鲜4真实使用用户1_334'
rpt.summary_chart(data,code,filename=filename)

# 交叉统计
cross_class='Q7'
filename='Q7交叉分析'
cross_qlist=[]
cross_order=[]
save_dstyle=['TWI','FO','TGI']
rpt.cross_chart(data,code,cross_class,filename=filename, cross_qlist=cross_qlist,\
cross_order=cross_order,save_dstyle=save_dstyle)
