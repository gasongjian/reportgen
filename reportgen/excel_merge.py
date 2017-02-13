# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 00:25:24 2017

@author: gason
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import report as rpt
from imp import reload
#reload(rpt)

w=pd.ExcelWriter(u'合并.xlsx')
# 合并EXCEL文件
f=pd.ExcelFile('.\\out\\性别差异分析_百分比表.xlsx')
row=0
for qq in f.sheet_names:
    d=pd.read_excel(f,qq,index_col=0,has_index_names=True)
    d.to_excel(w,startrow=row,index_label=True)
    row+=d.shape[0]+2

w.save()