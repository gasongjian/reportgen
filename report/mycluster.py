# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:51:00 2017

@author: gason
"""

# 聚类分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()

import report as rpt
from imp import reload
reload(rpt)


#  数据d导入
data0=rpt.read_data('data_cn.xlsx')
code=rpt.read_code('code_en.xlsx')
data=data0[data0['Q2'].notnull()]




'''
# 部分数据处理
dd=rpt.sa_to_ma(data['Q34'])
dd=dd.rename(columns={1:'Q34_A1',2:'Q34_A2',3:'Q34_A3',4:'Q34_A4',5:'Q34_A5',6:'Q34_A6',7:'Q34_A7'})
for i in range(7):
    data['Q34_A%d'%(i+1)]=dd['Q34_A%d'%(i+1)]

code['Q34a']={}
code['Q34a']['code']={'Q34_A1':code['Q34']['code'][1],\
'Q34_A2':code['Q34']['code'][2],'Q34_A3':code['Q34']['code'][3],\
'Q34_A4':code['Q34']['code'][4],'Q34_A5':code['Q34']['code'][5],\
'Q34_A6':code['Q34']['code'][6],'Q34_A7':code['Q34']['code'][7]}
code['Q34a']['qlist']=['Q34_A1','Q34_A2','Q34_A3','Q34_A4','Q34_A5','Q34_A6','Q34_A7']

#
data['Q36']=data['Q36']/10.0
data['Q37'].replace({6:5},inplace=True)
data['Q37']=data['Q37']/5.0
data['Q39'].replace({8:2,7:0},inplace=True)
data['Q39']=data['Q39']/6.0

#['Q36','Q37','Q39']+\
qlist_Q10=['Q10_A1','Q10_A5','Q10_A6','Q10_A19']
qlist=qlist_Q10+\
code['Q32']['qlist']+code['Q33']['qlist']+code['Q34a']['qlist']
#+code['Q34a']['qlist']



code_X=code['Q32']['code'].copy()
code_X.update(code['Q33']['code_r'])
code_X.update(code['Q34a']['code'])
code_X.update(code['Q10']['code'])
code_X.update({'Q36':u'年龄','Q37':u'学历','Q39':u'收入'})
'''



qlist=code['Q32']['qlist']
X=data[qlist]
X=pd.DataFrame(X.as_matrix(),columns=qlist)
X.fillna(0,inplace=True)


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.decomposition import PCA
import prince



est = KMeans(3)  # 4 clusters
est.fit(X)
labels=pd.Series(est.labels_)



X_PCA = PCA(2).fit_transform(X)
kwargs = dict(cmap = plt.cm.get_cmap('rainbow', 10),
              edgecolor='none', alpha=0.6)
plt.scatter(X_PCA[:, 0], X_PCA[:, 1], c=labels, **kwargs)

print(labels.value_counts())
X1=X.copy()
X1['labels']=labels
t=X1.groupby(['labels']).mean()
ca=prince.CA(t)
ca.plot_rows_columns(show_row_labels=True,show_column_labels=True)


data=pd.DataFrame(data.as_matrix(),columns=data.columns)
data['Q41']=labels

'''
X1=X.copy()
#X1=X1.join(data[code['Q34a']['qlist']])
X1=X1.join(data[['Q36','Q37','Q39']])

X1[u'labels']=labels
t=X1.groupby(['labels']).sum()
print(t)
t1=t[code['Q32']['qlist']]
t2=t[code['Q33']['qlist']]
t3=t[code['Q34a']['qlist']]
t4=t[qlist_Q10]
cdata=rpt.contingency(t1.T)
c1=cdata['CHI']
print(cdata['summary']['summary'])
cdata=rpt.contingency(t2.T)
c2=cdata['CHI']
print(cdata['summary']['summary'])
cdata=rpt.contingency(t3.T)
c3=cdata['CHI']
print(cdata['summary']['summary'])
cdata=rpt.contingency(t4.T)
c4=cdata['CHI']
print(cdata['summary']['summary'])
c=c1.copy().T
c=c.join(c2.T)
c=c.join(c3.T)
c=c.join(c4.T)
c=c.join(t[['Q36','Q37','Q39']])

t.rename(columns=code_X,inplace=True)
c.rename(columns=code_X,inplace=True)
t=t.T
c=c.T
t.to_excel('聚类分析51_fo.xls')
c.to_excel('聚类分析51_chi.xls')
X1.to_excel('聚类分析51_结果.xls')
#t.rename(columns=code_X,inplace=True)

'''

'''
0更喜欢Q32_A5;
1更喜欢Q32_A6,Q32_A8;
2更喜欢Q32_A5;
4更喜欢Q32_A7;
5更喜欢Q32_A3;

0更喜欢Q33_R2,Q33_R3,Q33_R4;
2更喜欢Q33_R8;
3更喜欢Q33_R9;
5更喜欢Q33_R1,Q33_R6,Q33_R7;

0更喜欢Q34_A4;
1更喜欢Q34_A2;
2更喜欢Q34_A2;
3更喜欢Q34_A3;
4更喜欢Q34_A1,Q34_A5,Q34_A6;
5更喜欢Q34_A4;
'''









