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
#import seaborn as sns; sns.set()

import report as rpt
from imp import reload
reload(rpt)


#  数据d导入
code=rpt.read_code('code.xlsx')
data=pd.read_excel('data.xlsx')

data['Q21a']=data[code['Q21']['qlist']].T.max().T
code['Q21a']={'content':'拍照频率','qtype':'单选题','qlist':['Q21a'],'code':{1: '几乎不使用', 2: '偶尔用', 3: '经常用'}}



from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.decomposition import PCA
import prince


qqlist=['Q11']
qqlist=['Q1','Q3','Q9','Q11','Q16','Q17','Q19','Q21a','Q25','Q34','Q35','Q36','Q37','Q39c']
qqlist1=[]
for q in qqlist:
    qqlist1+=code[q]['qlist']
qqlist1=qqlist1[:-1]


X=data[qqlist1]
X=X[X.T.notnull().all()]
#X_index=X.index
X=pd.DataFrame(X.as_matrix(),columns=qqlist1)
X=X.fillna(0)

# 分类
est = KMeans(3)  # 4 clusters
est.fit(X)
labels=pd.Series(est.labels_)

# 样本可视化
X_PCA = PCA(2).fit_transform(X)
kwargs = dict(cmap = plt.cm.get_cmap('rainbow', 10),
              edgecolor='none', alpha=0.6)
plt.scatter(X_PCA[:, 0], X_PCA[:, 1], c=labels, **kwargs)



print(labels.value_counts())
X1=X.copy()
X1['labels']=labels
t=X1.groupby(['labels']).mean()
#t=t.rename(columns=code['Q11']['code'])
ca=prince.CA(t)
ca.plot_rows_columns(show_row_labels=True,show_column_labels=True)












