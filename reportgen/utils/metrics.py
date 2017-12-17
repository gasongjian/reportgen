# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils.multiclass import type_of_target
#from collections import Iterable

'''
# 测试数据集
np.random.seed(100000)
# 非数值数据
X=pd.Series(np.random.choice(['a','b','c'],p=[0.2,0.3,0.5],size=1000))
X2=pd.Series(np.random.normal(10,5,size=1000))
# 数值数据
y=pd.Series(np.random.choice(['g','b'],p=[0.7,0.3],size=1000))
y1=pd.Series(np.random.choice(['g','b','n'],p=[0.6,0.3,0.1],size=1000))
# 评分卡预测概率数据
xp=np.zeros(1000)
t1=np.random.normal(0.3,0.1,size=(y=='g').sum())
t1[t1<0]=0
t1[t1>1]=1
t2=np.random.normal(0.7,0.1,size=(y=='b').sum())
t2[t2<0]=0
t2[t2>1]=1
xp[y=='g']=t1
xp[y=='b']=t2
xp=pd.Series(xp)
'''


'''
模型统一形式：
parameter
--------
X,模型输入数据,且只处理一个特征
y,模型输出数据，可以缺省
bins,连续数据离散化参数,缺省auto

return
------
h
'''



"""计算Woe
==============

针对离散数据,我们需要将离散的枚举值替换成数值才可以用于计算.这些数值就是各个枚举值的权重.
广义上讲,woe可以算是一种编码方式.
如何计算这些权重呢?这就得训练.我们需要一组二值代表签数据,
通过统计离散特征不同枚举值对目标数据的响应情况来计算触发概率,


触发概率
---------

本模型针对二分类问题,事件也就是只有True,False两种.我们当然认为True,枚举值i的触发概率可以这样计算,
比如某个枚举值i对true的触发概率,就是所有i值时是true的数量除以总的t的数量

.. math:: p_f = {\frac {f_i} {f_{total}}}

.. math:: p_t = {\frac {t_i} {t_{total}}}


woe
------

woe就是计算正负概率的信息值

.. math:: woe_i = log_2({\frac {p_f} {p_t}})


iv
-----

iv值就是这以特征的总信息量也就是各枚举值信息量的和

.. math:: IV_i = (p_f-p_t)*log_2({\frac {p_f} {p_t}})

.. math:: IV = \sum_{k=0}^n IV_i


使用方法:
----------

>>> from sklearn import datasets
>>> import pandas as pd
>>> iris = datasets.load_iris()
>>> y = iris.target
>>> z = (y==0)
>>> l = pd.DataFrame(iris.data,columns=iris.feature_names)
>>> d = Discretization([0,5.0,5.5,7])
>>> re = d.transform(l["sepal length (cm)"])
>>> woe = WeightOfEvidence()
>>> woe.fit(re,z)
>>> woe.woe
{'(0, 5]': 2.6390573296152589,
 '(5, 5.5]': 1.5581446180465499,
 '(5.5, 7]': -2.5389738710582761}
>>> woe.iv
3.617034906554693
>>> test = ['(0, 5]', '(0, 5]', '(5.5, 7]', '(5.5, 7]', '(5, 5.5]', '(5, 5.5]',
       '(5.5, 7]', '(5, 5.5]', '(5, 5.5]', '(5, 5.5]', '(0, 5]']
>>> woe.transform(test)
array([ 2.63905733,  2.63905733, -2.53897387, -2.53897387,  1.55814462,
        1.55814462, -2.53897387,  1.55814462,  1.55814462,  1.55814462,
        2.63905733])

"""




class WeightOfEvidence():
    """计算某一离散特征的woe值
    Attributes:
        woe (Dict): - 训练好的证据权重
        iv (Float): - 训练的离散特征的信息量
    """

    def __init__(self):
        self.woe = None
        self.iv = None

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
        x_labels = np.unique(x)
        pos_dic = {}
        for x1 in x_labels:
            y1 = tag[np.where(x == x1)[0]]
            event_count = (y1 == event).sum()
            non_event_count = y1.shape[-1] - event_count
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            pos_dic[x1] = (rate_event, rate_non_event)
        return pos_dic

    def fit(self, x, y, event=1, woe_min=-20, woe_max=20):
        """训练对单独一项自变量(列,特征)的woe值.
        Parameters:
        -----------
            x (Sequence): - 离散特征序列
            y (Sequence): - 用于训练的标签序列
            event (any): - True指代的触发事件
            woe_min (munber): - woe的最小值,默认值为-20
            woe_max (munber): - woe的最大值,默认值为20
        """
        woe_dict = {}
        iv = 0
        pos_dic = self._posibility(x=x, tag=y, event=event)
        for l, (rate_event, rate_non_event) in pos_dic.items():
            if rate_event == 0:
                woe1 = woe_min
            elif rate_non_event == 0:
                woe1 = woe_max
            else:
                woe1 = np.log(rate_event / rate_non_event)  # np.log就是ln
            iv += (rate_event - rate_non_event) * woe1
            woe_dict[l] = woe1
        self.woe = woe_dict
        self.iv = iv

    def transform(self, X):
        """将离散特征序列转换为woe值组成的序列
        Parameters:
            X (Sequence): - 离散特征序列
        Returns:
            numpy.array: - 替换特征序列枚举值为woe对应数值后的序列
        """
        return np.array([self.woe.get(i) for i in X])


class feature_encoder():
    '''
    用于单个特征对因变量的分析，如
    - 该特征中每个item的影响力
    - 对item重编码
 
    '''

    def chi2(X,y):
        N=pd.Series(y).count()
        fo=pd.crosstab(X,y)
        fe=stats.contingency.expected_freq(fo)
        weight_chi2=(fo-fe)**2/fe/N/min(fo.shape[0],fo.shape[1])
        weight_chi2=weight_chi2.sum(axis=1)
        return weight_chi2
    
    
    def woe(X,y):
        ctable=pd.crosstab(X,y)
        # 如果有0则每一项都加1
        ctable=ctable+1 if (ctable==0).any().any() else ctable
        if ctable.shape[1]==2:
            n_g,n_b=ctable.sum()
            ctable=(ctable/ctable.sum()).assign(woe=lambda x:np.log2(x.iloc[:,0]/x.iloc[:,1]))\
            .assign(ivi=lambda x:(x.iloc[:,0]-x.iloc[:,1])*x['woe'])
            return ctable.loc[:,['woe','ivi']]
        else:
            woe_dict={}
            p=ctable.sum()/ctable.sum().sum()
            for cc in ctable.columns:
                ctable_bin=pd.DataFrame(index=ctable.index,columns=['one','rest'])
                ctable_bin['one']=ctable.loc[:,cc]
                ctable_bin['rest']=ctable.loc[:,~(ctable.columns==cc)].sum(axis=1)
                n_o,n_r=ctable_bin.sum()
                ctable_bin=ctable_bin/ctable_bin.sum()
                ctable_bin['woe']=np.log2(ctable_bin['one']/ctable_bin['rest'])
                ctable_bin['ivi']=(ctable_bin['one']-ctable_bin['rest'])*ctable_bin['woe']
                woe_dict[cc]=ctable_bin.loc[:,['woe','ivi']]
            tmp=0
            for cc in ctable.columns:
                tmp+=woe_dict[cc]*p[cc]
            woe_dict['avg']=tmp
            return woe_dict        




def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From http://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    iqr = stats.scoreatpercentile(a, 75)-stats.scoreatpercentile(a, 25)   
    h = 2*iqr/(len(a)**(1/3))
    bins=int(np.ceil((a.max()-a.min())/h)) if h!=0 else int(np.sqrt(a.size))
    return bins



def chisquare(X,y):
    '''
    计算一组数据的卡方值
    '''
    chi2_value=pd.Series(index=X.columns)
    chi2_pvalue=pd.Series(index=X.columns)
    for c in X.columns:
        fo=pd.crosstab(X[c],y)
        s=stats.chi2_contingency(fo)
        chi2_value[c]=s[0]
        chi2_pvalue[c]=s[1]
    return (chi2_value,chi2_pvalue)



def info_value(X,y,bins='auto'):
    '''计算连续变量的IV值
    计算X和y之间的IV值
    IV=\sum (g_k/n_g-b_k/n_b)*log2(g_k*n_b/n_g/)
    '''
    threshold=[]
    for q in [0.05,0.04,0.03,0.02,0.01,1e-7]:
         t_down=max([X[y==k].quantile(q) for k in y.dropna().unique()])
         t_up=min([X[y==k].quantile(1-q) for k in y.dropna().unique()])
         threshold.append((t_down,t_up))
  
    if bins is not None:
        X=pd.cut(X,bins)
    ctable=pd.crosstab(X,y)
    p=ctable.sum()/ctable.sum().sum()   
    if ctable.shape[1]==2:
        ctable=ctable/ctable.sum()
        IV=((ctable.iloc[:,0]-ctable.iloc[:,1])*np.log2(ctable.iloc[:,0]/ctable.iloc[:,1])).sum()
        return IV
    
    IV=0
    for cc in ctable.columns:
        ctable_bin=pd.concat([ctable[cc],ctable.loc[:,~(ctable.columns==cc)].sum(axis=1)],axis=1)
        ctable_bin=ctable_bin/ctable_bin.sum()
        IV_bin=((ctable_bin.iloc[:,0]-ctable_bin.iloc[:,1])*np.log2(ctable_bin.iloc[:,0]/ctable_bin.iloc[:,1])).sum()
        IV+=IV_bin*p[cc]
    return IV




class entropy():
    
    '''
    计算样本的熵以及相关的指标
    函数的输入默认均为原始的样本集
    
    '''
    def entropy(X,dtype='category'):
        '''
        计算随机变量的信息熵
        H(X)=-\sum p_i log2(p_i)
        '''       
        X=pd.Series(X)
        
        if dtype == 'auto':
            if len(X.dropna().unique())>min(40,np.sqrt(X.dropna().count())):
                dtype='numeric'
            else:
                dtype='category'
       
        if dtype == 'category':
            p=X.value_counts(normalize=True)
        elif dtype == 'numeric':
            bins=_freedman_diaconis_bins(X)
            p=X.value_counts(normalize=True,bins=bins)
        elif (X>=0).all() and X.sum()==1:
            p=X
        else:
            return None     
        p=p[p>0]
        h=-(p*np.log2(p)).sum()          
        return h


    def entropy_condition(X,y,dtype_x='category',dtype_y='category'):
        '''
        计算随机变量的条件熵
        y必须是因子型变量
        H(X,y)=\sum p(y_i)H(X|y=y_i)
        '''
        #h=entropy_combination(X,y)-entropy(y)
        y=pd.Series(y)
        p=y.value_counts(normalize=True)
        h=0
        for yi in y.dropna().unique():
            h+=p[yi]*entropy.entropy(X[y==yi],dtype=dtype_x)
        return h

    def entropy_combination(X,y,dtype_x='category',dtype_y='category'):
        '''
        计算随机变量的联合熵
        H(X,y)=-\sum p(x_i,y_i)*log2(p(x_i,y_i))=H(X)+H(y|X)
        '''
        '''
        w=pd.crosstab(X,y)
        N=w.sum().sum()
        w=w/N
        w=w.values.flatten()
        w=w[w>0]
        h=-(w*np.log2(w)).sum()
        '''
        h=entropy.entropy(y,dtype_y)+entropy.entropy_condition(X,y,dtype_x)
        return h

    def mutual_info(X,y,dtype_x='category'):
        ''' 
        计算随机变量的互信息
        I(X;y)=H(X)-H(X|y)=H(y)-H(y|X)
        '''
        h=entropy.entropy(X,dtype_x)-entropy.entropy_condition(X,y,dtype_x)
        return h
    
    def info_gain(X,y,dtype_x='category'):
        ''' 
        计算随机变量的互信息
        I(X;y)=H(X)-H(X|y)=H(y)-H(y|X)
        '''
        h=entropy.entropy(X,dtype_x)-entropy.entropy_condition(X,y,dtype_x)
        return h
    
    def info_gain_ratio(X,y,dtype_x='category'):
        ''' 
        计算随机变量的信息增益比，此时X是总体，y是某个特征
        I(X;y)=H(X)-H(X|y)=H(y)-H(y|X)
        IG(X;y)=I(X;y)/H(y)
        '''
        h=entropy.entropy(X,dtype_x)-entropy.entropy_condition(X,y,dtype_x)
        hy=entropy.entropy(y)
        h=h/hy if hy>0 else 0
        return h
           
    
    
    def entropy_cross(X,y,dtype='category'):
        '''
        计算随机变量的交叉熵
        要求X和y的测度空间相同,此时X和y的样本数量可以不一致
        
        H(p,q)=-\sum p(x)log2(q(x))
        
        parameter
        --------
        dtype: X和y的数据类型，因子变量category和数值变量numeric，默认是category
        '''
        X=pd.Series(X)
        y=pd.Series(y)
        if dtype=='category':
            p=X.value_counts(normalize=True)
            q=y.value_counts(normalize=True)
            h=-(p*np.log2(q)).sum()
        elif dtype=='numeric':
            bins_X=_freedman_diaconis_bins(X)
            bins_y=_freedman_diaconis_bins(y)
            p=X.value_counts(normalize=True,bins=bins_X)
            q=y.value_counts(normalize=True,bins=bins_y)
            h=-(p*np.log2(q)).sum()
        return h


    def entropy_relative(X,y,dtype='category'):
        '''
        计算随机变量的相对熵
        要求X和y的测度空间相同,此时X和y的样本数量可以不一致
        D=\sum p(x) log2(p(x)/q(x))=H(p,q)-H(p)
        
        parameter
        --------
        dtype: X和y的数据类型，因子变量category和数值变量numeric，默认是category
        '''      

        X=pd.Series(X)
        y=pd.Series(y)
        if dtype=='category':
            p=X.value_counts(normalize=True)
            q=y.value_counts(normalize=True)
            #h=entropy.entropy_cross(p,q)-entropy.entropy(p)
            h=(p*np.log2(p/q)).sum()
        elif dtype=='numeric':
            bins_X=_freedman_diaconis_bins(X)
            bins_y=_freedman_diaconis_bins(y)
            p=X.value_counts(normalize=True,bins=bins_X)
            q=y.value_counts(normalize=True,bins=bins_y)
            h=(p*np.log2(p/q)).sum()

        return h
