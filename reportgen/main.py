# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 11:36:51 2016
@author: gason
"""
import re
import os
from math import *
import pandas as pd
import numpy as np
import report as rpt

from pptx import Presentation



# =================[数据导入]==========================
data=pd.read_csv('data.csv')
code=rpt.read_code('code.xlsx')




cross_list=list(sorted(code,key=lambda c:int(re.findall('\d+',c)[0])))
cross_class='Q24'
cross_list.remove(cross_class)
filename='russia'
if not os.path.exists('.\\out'):
    os.mkdir('.\\out')
#Writer=pd.ExcelWriter('.\\out\\'+filename+'.xlsx')
Writer2=pd.ExcelWriter('.\\out\\'+filename+'_chi.xlsx')
cross_qlist=code[cross_class]['qlist']
for qq in cross_list:    
    if code[qq]['qtype'] == u'单选题':
        t,t1=rpt.rptcrosstab(data[cross_qlist],data[qq],code[cross_class])
        t.rename(columns=code[qq]['code'],inplace=True)
        t1.rename(columns=code[qq]['code'],inplace=True)
        t.fillna(0,inplace=True)
        t1.fillna(0,inplace=True)
        t1.drop(u'总体',axis=1,inplace=True)
        fe=t1.copy()
        N=fe.sum().sum()
        for i in fe.index:
            for j in fe.columns:
                fe.loc[i,j]=fe.loc[i,:].sum()*fe.loc[:,j].sum()/N
        TSI=t1-fe
        chi=(t1-fe)*(t1/fe-1)*(TSI.applymap(lambda x: int(x>0))*2-1)
       

        #for c in t.columns:
        #    t1[c]=t[c]/t[u'总体']
        #t1.loc[u'sign']=t1.mean()+2*t1.std()
        #t.loc[u'sign']=t1.loc[u'sign'].copy()
        #t2=pd.concat([t,t1],axis=1)
        #t2.to_excel(Writer,qq)
        chi.to_excel(Writer2,qq)
#Writer.save()
Writer2.save()
