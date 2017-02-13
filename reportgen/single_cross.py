# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 22:05:02 2017

@author: gason
"""

import report as rpt
import pandas as pd

tchi=pd.DataFrame()
f100=pd.DataFrame()
fre=pd.DataFrame()

t,t1=rpt.crosstab(data_index,data_column,code_index,code_column)
cdata=rpt.contingency(t1)
CHI=cdata['CHI']
tchi.join(CHI)
f100.join(t)
fre.join(t1)



