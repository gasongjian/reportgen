# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
import random

#from collections import Iterable

__all__=['entropy',
'entropyc',
'entropyd',
'chi2',
'info_value']


# 待定，还未修改好
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



def chi2(X,y):
    '''计算一组数据的卡方值，弥补sklearn中的chi2只支持2*2的缺憾
    parameter
    ----------
    X:可以是单个特征，也可以是一组特征
    y:目标变量
    
    return
    ------
    chi2_value: np.array 数组
    chi2_pvalue：np.array 数组
    '''
    X=np.asarray(X)
    if len(X.shape)==1:
        X=X.reshape((len(X),1))
    X=pd.DataFrame(X)
    chi2_value=[]
    chi2_pvalue=[]
    for c in X.columns:
        fo=pd.crosstab(X[c],y)
        s=stats.chi2_contingency(fo)
        chi2_value.append(s[0])
        chi2_pvalue.append(s[1])
    return (np.array(chi2_value),np.array(chi2_pvalue))



# 待定
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



# 计算离散随机变量的熵
class entropy:

    '''
    计算样本的熵以及相关的指标
    函数的输入默认均为原始的样本集

    '''
    def entropy(X):
        '''
        计算随机变量的信息熵
        H(X)=-\sum p_i log2(p_i)
        '''
        X=pd.Series(X)
        p=X.value_counts(normalize=True)
        p=p[p>0]
        h=-(p*np.log2(p)).sum()
        return h


    def cond_entropy(x,y):
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
            h+=p[yi]*entropy.entropy(x[y==yi])
        return h

    def comb_entropy(x,y):
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
        h=entropy.entropy(y)+entropy.cond_entropy(x,y)
        return h

    def mutual_info(x,y):
        '''
        计算随机变量的互信息
        I(X;y)=H(X)-H(X|y)=H(y)-H(y|X)
        '''
        h=entropy.entropy(x)-entropy.cond_entropy(x,y)
        return h

    def info_gain(x,y):
        '''
        计算随机变量的互信息
        I(X;y)=H(X)-H(X|y)=H(y)-H(y|X)
        '''
        h=entropy.entropy(x)-entropy.cond_entropy(x,y)
        return h

    def info_gain_ratio(x,y):
        '''
        计算随机变量的信息增益比，此时X是总体，y是某个特征
        I(X;y)=H(X)-H(X|y)=H(y)-H(y|X)
        IG(X;y)=I(X;y)/H(y)
        '''
        h=entropy.entropy(x)-entropy.cond_entropy(x,y)
        hy=entropy.entropy(y)
        h=h/hy if hy>0 else 0
        return h



    def cross_entropy(x,y):
        '''
        计算随机变量的交叉熵
        要求X和y的测度空间相同,此时X和y的样本数量可以不一致

        H(p,q)=-\sum p(x)log2(q(x))

        parameter
        --------
        '''
        X=pd.Series(x)
        y=pd.Series(y)
        p=X.value_counts(normalize=True)
        q=y.value_counts(normalize=True)
        h=-(p*np.log2(q)).sum()
        return h


    def relative_entropy(x,y):
        '''
        计算随机变量的相对熵
        要求X和y的测度空间相同,此时X和y的样本数量可以不一致
        D=\sum p(x) log2(p(x)/q(x))=H(p,q)-H(p)

        parameter
        --------
        dtype: X和y的数据类型，因子变量category和数值变量numeric，默认是category
        '''

        X=pd.Series(x)
        y=pd.Series(y)
        p=X.value_counts(normalize=True)
        q=y.value_counts(normalize=True)
        #h=entropy.entropy_cross(p,q)-entropy.entropy(p)
        h=(p*np.log2(p/q)).sum()
        return h




# 计算连续变量的熵（利用分布进行近似 CONTINUOUS ESTIMATORS）
class entropyc:

    '''
    原作者：Greg Ver Steeg
    GitHub：https://github.com/gregversteeg/NPEET
    Or go to http://www.isi.edu/~gregv/npeet.html

    ref:Alexander Kraskov etc. Estimating mutual information. Phys. Rev. E, 69:066138, Jun 2004

    连续分布的熵估计
    '''
    
    def __reshape(x):
        x=np.asarray(x)
        if len(x.shape)==1:
            x=x.reshape((len(x),1))
        return x

    def entropy(x, k=3, base=2):
        """
        The classic K-L k-nearest neighbor continuous entropy estimator

        if x is a one-dimensional scalar and we have:
        H(X)=-\sum p_i log2(p_i)
        if we only have random sample (x1 . . . xN) of N realizations of X,
        we can estimator H(X):

        H(X) = −ψ(k) + ψ(N) + \log c_d + d/N \sum_{i=1}^{N} \log eps(i)

        where ψ(x) is digammer funciton,d is the dimention of x,
         c_d is the volume of the d-dimensional unit ball
        eps(i) is twice the distance from xi to its k-th neighbour

        parameter
        ---------
        x: 某个分布的抽样，且支持多维。
        k: k近邻的
        base：2

        return
        -------
        entropy
        """
        x=entropyc.__reshape(x)
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        d = len(x[0])
        N = len(x)
        intens = 1e-10  # small noise to break degeneracy, see doc.
        x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
        tree = ss.cKDTree(x)
        nn = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in x]
        const = digamma(N) - digamma(k) + d * log(base)
        return (const + d * np.mean(list(map(log, nn)))) / log(base)

    def cond_entropy(x, y, k=3, base=2):
      """ The classic K-L k-nearest neighbor continuous entropy estimator for the
          entropy of X conditioned on Y.
      """
      hxy = entropyc.entropy([xi + yi for (xi, yi) in zip(x, y)], k, base)
      hy = entropyc.entropy(y, k, base)
      return hxy - hy

    def __column(xs, i):
      return [[x[i]] for x in xs]

    def tc(xs, k=3, base=2):
      xis = [entropyc.entropy(entropyc.__column(xs, i), k, base) for i in range(0, len(xs[0]))]
      return np.sum(xis) - entropyc.entropy(xs, k, base)

    def ctc(xs, y, k=3, base=2):
      xis = [entropyc.cond_entropy(entropyc.__column(xs, i), y, k, base) for i in range(0, len(xs[0]))]
      return np.sum(xis) - entropyc.cond_entropy(xs, y, k, base)

    def corex(xs, ys, k=3, base=2):
      cxis = [entropyc.mutual_info(entropyc.__column(xs, i), ys, k, base) for i in range(0, len(xs[0]))]
      return np.sum(cxis) - entropyc.mutual_info(xs, ys, k, base)

    def mutual_info(x, y, k=3, base=2):
        """ Mutual information of x and y
            x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        x=entropyc.__reshape(x)
        y=entropyc.__reshape(y)
        assert len(x) == len(y), "Lists should have same length"
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        intens = 1e-10  # small noise to break degeneracy, see doc.
        x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
        y = [list(p + intens * nr.rand(len(y[0]))) for p in y]
        points = zip2(x, y)
        # Find nearest neighbors in joint space, p=inf means max-norm
        tree = ss.cKDTree(points)
        dvec = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
        return (-a - b + c + d) / log(base)


    def cond_mutual_info(x, y, z, k=3, base=2):
        """ Mutual information of x and y, conditioned on z
            x, y, z should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        x=entropyc.__reshape(x)
        y=entropyc.__reshape(y)
        z=entropyc.__reshape(z)
        assert len(x) == len(y), "Lists should have same length"
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        intens = 1e-10  # small noise to break degeneracy, see doc.
        x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
        y = [list(p + intens * nr.rand(len(y[0]))) for p in y]
        z = [list(p + intens * nr.rand(len(z[0]))) for p in z]
        points = zip2(x, y, z)
        # Find nearest neighbors in joint space, p=inf means max-norm
        tree = ss.cKDTree(points)
        dvec = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in points]
        a, b, c, d = avgdigamma(zip2(x, z), dvec), avgdigamma(zip2(y, z), dvec), avgdigamma(z, dvec), digamma(k)
        return (-a - b + c + d) / log(base)


    def kl_div(x, xp, k=3, base=2):
        """ KL Divergence between p and q for x~p(x), xp~q(x)
            x, xp should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        """
        x=entropyc.__reshape(x)
        xp=entropyc.__reshape(xp)
        assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
        assert k <= len(xp) - 1, "Set k smaller than num. samples - 1"
        assert len(x[0]) == len(xp[0]), "Two distributions must have same dim."
        d = len(x[0])
        n = len(x)
        m = len(xp)
        const = log(m) - log(n - 1)
        tree = ss.cKDTree(x)
        treep = ss.cKDTree(xp)
        nn = [tree.query(point, k + 1, p=float('inf'))[0][k] for point in x]
        nnp = [treep.query(point, k, p=float('inf'))[0][k - 1] for point in x]
        return (const + d * np.mean(list(map(log, nnp))) - d * np.mean(list(map(log, nn)))) / log(base)



# 计算随机变量的熵（直接离散话估计 DISCRETE ESTIMATORS）
class entropyd:

    def entropy(sx, base=2):
        """ Discrete entropy estimator
            Given a list of samples which can be any hashable object
        """
        return entropyd.entropyfromprobs(entropyd.hist(sx), base=base)


    def mutual_info(x, y, base=2):
        """ Discrete mutual information estimator
            Given a list of samples which can be any hashable object
        """
        return -entropyd.entropy(zip(x, y), base) + entropyd.entropy(x, base) + entropyd.entropy(y, base)

    def cond_mutual_info(x, y, z):
        """ Discrete mutual information estimator
            Given a list of samples which can be any hashable object
        """
        return entropyd.entropy(zip(y, z))+entropyd.entropy(zip(x, z))-entropyd.entropy(zip(x, y, z))-entropyd.entropy(z)

    def cond_entropy(x, y, base=2):
      """ The classic K-L k-nearest neighbor continuous entropy estimator for the
          entropy of X conditioned on Y.
      """
      return entropyd.entropy(zip(x, y), base) - entropyd.entropy(y, base)

    def tcd(xs, base=2):
      xis = [entropyd.entropy(entropyd._column(xs, i), base) for i in range(0, len(xs[0]))]
      hx = entropyd.entropy(xs, base)
      return np.sum(xis) - hx

    def ctcd(xs, y, base=2):
      xis = [entropyd.cond_entropy(entropyd._column(xs, i), y, base) for i in range(0, len(xs[0]))]
      return np.sum(xis) - entropyd.cond_entropy(xs, y, base)

    def corexd(xs, ys, base=2):
      cxis = [entropyd.mutual_infod(entropyd._column(xs, i), ys, base) for i in range(0, len(xs[0]))]
      return np.sum(cxis) - entropyd.mutual_info(xs, ys, base)

    def hist(sx):
        sx = discretize(sx)
        # Histogram from list of samples
        d = dict()
        for s in sx:
            if type(s) == list:
              s = tuple(s)
            d[s] = d.get(s, 0) + 1
        return map(lambda z: float(z) / len(sx), d.values())


    def entropyfromprobs(probs, base=2):
        # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
        return -sum(map(entropyd.elog, probs)) / log(base)

    def _column(xs, i):
      return [[x[i]] for x in xs]

    def elog(x):
        # for entropy, 0 log 0 = 0. but we get an error for putting log 0
        if x <= 0. or x >= 1.:
            return 0
        else:
            return x * log(x)





# UTILITY FUNCTIONS
def vectorize(scalarlist):
    """ Turn a list of scalars into a list of one-d vectors
    """
    return [[x] for x in scalarlist]


def shuffle_test(measure, x, y, z=False, ns=200, ci=0.95, **kwargs):
    """ Shuffle test
        Repeatedly shuffle the x-values and then estimate measure(x, y, [z]).
        Returns the mean and conf. interval ('ci=0.95' default) over 'ns' runs.
        'measure' could me mi, cmi, e.g. Keyword arguments can be passed.
        Mutual information and CMI should have a mean near zero.
    """
    xp = x[:]  # A copy that we can shuffle
    outputs = []
    for i in range(ns):
        random.shuffle(xp)
        if z:
            outputs.append(measure(xp, y, z, **kwargs))
        else:
            outputs.append(measure(xp, y, **kwargs))
    outputs.sort()
    return np.mean(outputs), (outputs[int((1. - ci) / 2 * ns)], outputs[int((1. + ci) / 2 * ns)])

def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From http://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    iqr = stats.scoreatpercentile(a, 75)-stats.scoreatpercentile(a, 25)
    h = 2*iqr/(len(a)**(1/3))
    bins=int(np.ceil((a.max()-a.min())/h)) if h!=0 else int(np.sqrt(a.size))
    return bins

# INTERNAL FUNCTIONS

def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    N = len(points)
    tree = ss.cKDTree(points)
    avg = 0.
    for i in range(N):
        dist = dvec[i]
        # subtlety, we don't include the boundary point,
        # but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(points[i], dist - 1e-15, p=float('inf')))
        avg += digamma(num_points) / N
    return avg


def zip2(*args):
    # zip2(x, y) takes the lists of vectors and makes it a list of vectors in a joint space
    # E.g. zip2([[1], [2], [3]], [[4], [5], [6]]) = [[1, 4], [2, 5], [3, 6]]
    return [sum(sublist, []) for sublist in zip(*args)]

def discretize(xs):
    def discretize_one(x):
        if len(x) > 1:
            return tuple(x)
        else:
            return x[0]
    # discretize(xs) takes a list of vectors and makes it a list of tuples or scalars
    return [discretize_one(x) for x in xs]
