"""
离散化连续数据

有监督方法：
ChiMerge
无监督方法：
等频
等距

"""

import numpy as np
import pandas as pd
from scipy import stats

def _chisqure_fo(fo):
    if any(fo==0):
        fo=fo+1
    s=stats.chi2_contingency(fo)
    return s[0],s[1]


def chimerge(x,y,max_intervals,threshold,sample=None):
    '''卡方分箱
    parameter
    ---------
    x: 特征变量
    y: 目标变量
    max_intervals: 最大的区间
    threshold：卡方阈值(两个变量)
    
    return
    ------
    bins:
    '''
    x=np.asarray(x).flatten()
    y=np.asarray(y).flatten()
    class_y=list(np.unique(y))
    value_max=x.max()
    #value_max=np.sort(x)[-1]
    value_min=x.min()
    if isinstance(sample,int):
        sample=min(sample,len(x))
        tmp=set()
        while tmp!=set(class_y):
            cc=np.random.choice([True,False],size=len(x),p=[sample/len(x),1-sample/len(x)])
            tmp=set(np.unique(y[cc]))
        x=x[cc]
        y=y[cc]
    fo=pd.crosstab(x,y)# 列联表
    fo=fo.sort_index()

    #fo_intervals=np.array(fo.index)
    
    while fo.shape[0] > max_intervals:
        chitest={}
        index=list(fo.index)
        for r in range(fo.shape[0]-1):
            #chi2,_=stats.chi2_contingency(fo.iloc[[r,r+1],:])
            chi2,_=_chisqure_fo(fo.iloc[[r,r+1],:])
            if chi2 not in chitest:
                chitest[chi2]=[]
            chitest[chi2].append((r,r+1))
        smallest = min(chitest.keys())
        if smallest <= threshold:
            #print('最小的chi2值: {}'.format(smallest))
            #print([(index[r[0]],index[r[1]]) for r in list(reversed(chitest[smallest]))])
            for (lower,upper) in list(reversed(chitest[smallest])):
                fo.loc[index[lower],:]=fo.loc[index[lower],:]+fo.loc[index[upper],:]
                fo = fo.drop(index[upper],axis=0)
                #print('已经删除 {}'.format(index[upper]))
        else:
            break
    bins=list(fo.index)+[value_max]
    bins[0]=value_min
    # 如果bins都是数值，则最左和最右都扩大1%以囊括最小最大值
    if np.issubdtype(type(bins[0]),np.number):
        bins[0]=bins[0]*0.99 if bins[0]>0 else bins[0]-0.01
        bins[-1]=bins[-1]*1.01
    return bins


class Discretization():
    """离散化连续数据.需要实例化以保存bins状态.
    parameter:
    bins (Sequence): - 用于分段的列表,第一位为下限,最后一位为上限
    method: 离散的方法
    """

    def __init__(self, bins=None,method='auto',**kwargs):
        self.bins = bins
        self.method=method
        self.labels=None
        if 'max_intervals' in kwargs:
            self.max_intervals=kwargs['max_intervals']
        else:
            self.max_intervals=10
        if 'threshold' in kwargs:
            self.threshold=kwargs['threshold']
        else:
            self.threshold=5
        if 'sample' in kwargs:
            self.sample=kwargs['sample']
        else:
            self.sample=None

    def fit(self,x,y=None):
        if self.method == 'auto':
            if y is not None:
                method='chimerge'
            elif self.bins is not None:
                method=''
            else:
                method=''
        else:
            method=self.method
            
        if method.lower() in ['chimerge']:
            bins=chimerge(x,y,max_intervals=self.max_intervals,threshold=self.threshold,sample=self.sample)
            self.bins=bins

    def transform(self, x):
        if isinstance(x,pd.core.series.Series):
            index=x.index
        else:
            index=None
        x=pd.Series(np.asarray(x).flatten(),index=index)
        s = pd.cut(x, bins=self.bins,right=False)
        s=s.map(lambda x:str(x))
        self.labels=list(s.cat.categories)
        return s

    def fit_transform(self,x,y=None):
        self.fit(x,y)
        return self.transform(x)