# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:09:46 2018

@author: JSong
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils.multiclass import type_of_target


__all__=['WeightOfEvidence',
         'chimerge',
         'Discretization']


def check_array(X,ensure_DataFrame=True,copy=True):
    '''Convert X to DataFrame   
    '''
    X=X.copy()
    if not(np.issubdtype(type(X),np.ndarray)):
        X=np.array(X)
    X=pd.DataFrame(X)
    return X


def _features_selected(X, selected="all"):
    """Apply a transform function to portion of selected features

    Parameters
    ----------
    X : {array-like}, shape [n_samples, n_features]
    
    selected: "all" or array of indices or mask
        Specify which features to apply the transform to.

    Returns
    -------
    n_features_new : array
    """

    X=check_array(X)
    if selected == "all":
        return np.array(X.columns)
    n_features = X.shape[1]
    sel = pd.Series(np.zeros(n_features, dtype=bool),index=X.columns)
    sel[np.asarray(selected)] = True
    return np.array(X.columns[sel])


class WeightOfEvidence():
    """ WOE Encoder
    
    parameters:
    -----------
    
    categorical_features : "all" or array of indices or mask
        Specify what features are treated as categorical.

        - 'all' (default): All features are treated as categorical.
        - array of indices: Array of categorical feature indices.
        - mask: Array of length n_features and with dtype=bool.
    encoder_na: default False, take nan as a single class of the features
        
    attribute:
    -----------
    woe (Dict): - the woe of trained data
    iv (Dict): - info value of trained data
    """

    def __init__(self,categorical_features='all',encoder_na=False,woe_min=-20, woe_max=20):
        self.woe = {}
        self.iv = {}
        self.encoder_na=encoder_na
        self.woe_min=woe_min
        self.woe_max=woe_max
        self.categorical_features=categorical_features

    def _posibility(self, x, tag, event=1):
        """计算触发概率
        Parameters:
        ----------
            x (Sequence): - 离散特征序列
            tag (Sequence): - 用于训练的标签序列
            event (any): - True指代的触发事件
        Returns:
        ----------
            Dict[str,Tuple[rate_T, rate_F]]: - 训练好后的好坏触发概率
        """
        if type_of_target(tag) not in ['binary']:
            raise AttributeError("tag must be a binary array")
        #if type_of_target(x) in ['continuous']:
        #    raise AttributeError("input array must not continuous")
        tag = np.array(tag)
        x = np.array(x)
        event_total = (tag == event).sum()
        non_event_total = tag.shape[-1] - event_total
        x_labels = pd.unique(x[pd.notnull(x)])
        pos_dic = {}
        for x1 in x_labels:
            # 当 x1 是nan时，y1 也为空
            y1 = tag[np.where(x == x1)[0]]
            event_count = (y1 == event).sum()
            non_event_count = y1.shape[-1] - event_count
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            pos_dic[x1] = (rate_event, rate_non_event)
        return pos_dic

    def fit(self, X, y, event=1):
        """训练对单独一项自变量(列,特征)的woe值.
        WOE_k=log (该特征中正类占比/该特征中负类占比)
        Parameters:
        -----------
            X : DataFrame, 训练数据
            y (Sequence):  标签
            event: - True指代的触发事件
            woe_min (munber): - woe的最小值,默认值为 -20
            woe_max (munber): - woe的最大值,默认值为 20
        """
        X = check_array(X,ensure_DataFrame=True)       
        y = np.array(y)       
        if np.isnan(y).sum()>0:
            raise AttributeError("y contain NaN number!")
        feartures_new=_features_selected(X,self.categorical_features)
        if self.encoder_na:
            X[feartures_new]=X[feartures_new].fillna('np.nan')
        for v in feartures_new:
            woe_dict = {}
            iv = 0
            pos_dic = self._posibility(x=X[v], tag=y, event=event)
            for l, (rate_event, rate_non_event) in pos_dic.items():
                if rate_event == 0:
                    woe1 = self.woe_min
                elif rate_non_event == 0:
                    woe1 = self.woe_max
                else:
                    woe1 = np.log(rate_event / rate_non_event)  # np.log就是ln
                iv += (rate_event - rate_non_event) * woe1
                woe_dict[l] = woe1
            self.woe[v] = woe_dict
            self.iv[v] = iv

    def transform(self, X):
        """将离散特征序列转换为woe值组成的序列
        Parameters:
            X : DataFrame, 训练数据
        Returns:
            DataFrame: - 替换特征序列枚举值为woe对应数值后的序列
        """
        X=check_array(X)
        feartures_new=_features_selected(X,self.categorical_features)
        if self.encoder_na:
            X[feartures_new]=X[feartures_new].fillna('np.nan')
        for v in feartures_new:
            X[v]=X[v].replace(self.woe[v])
        return X
    def fit_transform(self,X,y,event=1):
        self.fit(X, y, event=event)
        return self.transform(X)
        


def _chisqure_fo(fo):
    if any(fo==0):
        fo=fo+1
    s=stats.chi2_contingency(fo)
    return s[0],s[1]


def chimerge(x,y,max_intervals=30,threshold=5,sample=None):
    '''卡方分箱
    parameter
    ---------
    x: {array-like}, shape [n_samples, 1]
    y: target, connot contain nan 
    max_intervals: 最大的区间数
    threshold：卡方阈值(两个变量)
    sample: int,当样本数过大时，对数据进行取样
    
    return
    ------
    bins: 
    
    '''
    
    x=pd.Series(x)
    y=pd.Series(y)
    class_y=list(pd.unique(y[pd.notnull(y)]))
    value_max=x.max()
    #value_max=np.sort(x)[-1]
    value_min=x.min()
    # 随机取样，且确保取样后的y能包含class_y中的所有类别
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
   
    while fo.shape[0] > max_intervals:
        chitest={}
        index=list(fo.index)
        for r in range(len(fo)-1):
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

    def __init__(self, bins=None,method='auto',continous_features='all',**kwargs):
        self.bins = bins
        self.method=method
        self.continous_features=continous_features
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

    def fit(self,X,y=None):
        if self.method == 'auto':
            if y is not None:
                method='chimerge'
            elif self.bins is not None:
                method=''
            else:
                method=''
        else:
            method=self.method
        X=check_array(X)
        feartures_new=_features_selected(X,self.continous_features)           
        if method.lower() in ['chimerge']:
            self.bins={}
            for v in feartures_new:
                bins=chimerge(X[v],y,max_intervals=self.max_intervals,threshold=self.threshold,sample=self.sample)
                self.bins[v]=bins

    def transform(self, X):
        X=check_array(X)
        feartures_new=_features_selected(X,self.continous_features)
        for v in feartures_new:
            bins=self.bins[v]
            labels=['[{},{})'.format(bins[i],bins[i+1]) for i in range(len(bins)-1)]
            X[v] = pd.cut(X[v], bins=bins,labels=labels,right=False)
        return X

    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X)
        
        
        


