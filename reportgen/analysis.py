# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 21:53:32 2017

@author: gason
"""

import pandas as pd
import numpy as np
import re
import time
import os
from collections import Iterable

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_number
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_categorical_dtype
from scipy import stats
from sklearn import metrics

from . import report as _rpt
from . import config
from .report import genwordcloud
from .utils.metrics import entropyc

from .utils import iqr

#from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sns

_thisdir = os.path.split(__file__)[0]
# default chinese font
from matplotlib.font_manager import FontProperties
font_path=config.font_path
if font_path:
    myfont=FontProperties(fname=font_path)
    sns.set(font=myfont.get_name())


__all__=['type_of_var',
         'describe',
         'plot',
         'AnalysisReport',
         'ClassifierReport']


def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From http://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    h = 2 * iqr(a) / (len(a) ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((a.max() - a.min()) / h))



def distributions(a,hist=True,bins=None,norm_hist=True,kde=True,grid=None,gridsize=100,clip=None):
    '''数组的分布信息
    hist=True,则返回分布直方图(counts,bins)
    kde=True,则返回核密度估计数组(grid,y)

    example
    -------
    a=np.random.randint(1,50,size=(1000,1))
    '''
    a = np.asarray(a).squeeze()
    if hist:
        if bins is None:
            bins = min(_freedman_diaconis_bins(a), 50)
        counts,bins=np.histogram(a,bins=bins)
        if norm_hist:
            counts=counts/counts.sum()
    if kde:
        bw='scott'
        cut=3
        if clip is None:
            clip = (-np.inf, np.inf)
        try:
            kdemodel = stats.gaussian_kde(a, bw_method=bw)
        except TypeError:
            kdemodel = stats.gaussian_kde(a)
        bw = "scotts" if bw == "scott" else bw
        bw = getattr(kdemodel, "%s_factor" % bw)() * np.std(a)
        if grid is None:
            support_min = max(a.min() - bw * cut, clip[0])
            support_max = min(a.max() + bw * cut, clip[1])
            grid=np.linspace(support_min, support_max, gridsize)
        y = kdemodel(grid)
    if hist and not(kde):
        return counts,bins
    elif not(hist) and kde:
        return grid,y
    elif hist and kde:
        return ((counts,bins),(grid,y))
    else:
        return None


def dtype_detection(data,category_detection=True,StructureText_detection=True,\
datetime_to_category=True,criterion='sqrt',min_mean_counts=5,fix=False):
    '''检测数据中单个变量的数据类型
    将数据类型分为以下4种
    1. number,数值型
    2. category,因子
    3. datetime,时间类型
    4. text,文本型
    5. text_st,结构性文本，比如ID,
    6. group_number,连续

    parameter
    ---------
    data: pd.Series 数据, 仅支持一维
    # 如果有data,则函数会改变原来data的数据类型
    category_detection: bool,根据 nunique 检测是否是因子类型
    StructureText_detection: bool, 结构化文本，如列中都有一个分隔符"-"
    datetime_to_category: 时间序列如果 nunique过少是否转化成因子变量
    criterion: string or int, optional (default="sqrt",即样本数的开根号)
        支持：'sqrt'：样本量的开根号, int: 绝对数, 0-1的float：样本数的百分多少
        检测因子变量时，如果一个特征的nunique小于criterion,则判定为因子变量
    min_mean_counts: default 5,数值型判定为因子变量时，需要满足每个类别的平均频数要大于min_mean_counts
    fix: bool,是否返回修改好类型的数据
    

    return:
    result:dict{
        'name':列名,
        'vtype':变量类型,
        'ordered':是否是有序因子,
        'categories':所有的因子}

    '''
        
    assert len(data.shape)==1
    data=data.copy()
    data=pd.Series(data)
    dtype,name,n_sample=data.dtype,data.name,data.count()

    min_mean_counts=5
    if criterion=='sqrt':
        max_nuniques=np.sqrt(n_sample)
    elif isinstance(criterion,int):
        max_nuniques=criterion
    elif isinstance(criterion,float) and (0<criterion<1):
        max_nuniques=criterion
    else:
        max_nuniques=np.sqrt(n_sample)
    ordered=False
    categories=[]
    if is_numeric_dtype(dtype):
        vtype='number'
        ordered=False
        categories=[]
        # 纠正误分的数据类型。如将1.0，2.0，3.0都修正为1，2，3
        if data.dropna().astype(np.int64).sum()==data.dropna().sum():
            data[data.notnull()]=data[data.notnull()].astype(np.int64)
        if category_detection:
            nunique=len(data.dropna().unique())
            mean_counts=data.value_counts().median()
            if nunique<max_nuniques and mean_counts>=min_mean_counts:
                data=data.astype('category')
                ordered=data.cat.ordered
                vtype='category'
                categories=list(data.dropna().cat.categories)
        result={'name':name,'vtype':vtype,'ordered':ordered,'categories':categories}
    elif is_string_dtype(dtype):
        # 处理时间类型
        tmp=data.map(lambda x: np.nan if '%s'%x == 'nan' else len('%s'%x))
        tmp=tmp.dropna().astype(np.int64)       
        if not(any(data.dropna().map(is_number))) and 7<tmp.max()<20 and tmp.std()<0.1:
            try:
                data=pd.to_datetime(data)
            except :
                pass
        # 处理可能的因子类型
        #时间格式是否处理为True 且
        if datetime_to_category:
            if len(data.dropna().unique())<np.sqrt(n_sample):
                data=data.astype('category')
        else:
            nunique=len(data.dropna().unique())
            #print(data.dtype)
            if not(is_categorical_dtype(data.dtype)) and not(np.issubdtype(data.dtype,np.datetime64)) and nunique<max_nuniques:
                data=data.astype('category')

        # 在非因子类型的前提下，将百分数转化成浮点数，例如21.12%-->0.2112
        if is_string_dtype(data.dtype) and not(is_categorical_dtype(data.dtype)) and all(data.str.contains('%')):
            data=data.str.strip('%').astype(np.float64)/100

        if is_categorical_dtype(data.dtype):
            vtype='category'
            categories=list(data.cat.categories)
            ordered=data.cat.ordered
        # 时间格式
        elif np.issubdtype(data.dtype,np.datetime64):
            vtype='datetime'
        # 是否是结构化数组
        elif StructureText_detection and tmp.dropna().std()==0:
            # 不可迭代，不是字符串
            if not(isinstance(data.dropna().iloc[0],Iterable)):
                vtype='text'
            else:
                k=set(list(data.dropna().iloc[0]))
                for x in data:
                    if isinstance(x,str) and len(x)>0:
                        k&=set(list(x))
                if len(k)>0:
                    vtype='text_st'
                else:
                    vtype='text'
        elif is_numeric_dtype(data.dtype):
            vtype='number'
            ordered=False
            categories=[]
        else:
            vtype='text'
        result={'name':name,'vtype':vtype,'ordered':ordered,'categories':categories}
    elif is_datetime64_any_dtype(dtype):
        vtype='datetime'
        result={'name':name,'vtype':vtype,'ordered':ordered,'categories':categories}
    else:
        print('unknown dtype!')
        result=None
        
    if fix:
        return result,data
    else:
        return result



def type_of_var(data,category_detection=True,criterion='sqrt',min_mean_counts=5,copy=True):
    '''返回各个变量的类型
    将数据类型分为以下4种
    1. number,数值型
    2. category,因子
    3. datetime,时间类型
    4. text,文本型
    5. text_st,结构性文本，比如ID,

    parameters
    ----------
    data: pd.DataFrame类型
    category_detection: bool,根据 nunique 检测是否是因子类型
    criterion: string or int, optional (default="sqrt",即样本数的开根号)
        支持：'sqrt'：样本量的开根号, int: 绝对数, 0-1的float：样本数的百分多少
        检测因子变量时，如果一个特征的nunique小于criterion,则判定为因子变量
    min_mean_counts: default 5,数值型判定为因子变量时，需要满足每个类别的平均频数要大于min_mean_counts
    copy: bool, 是否更改数据类型，如时间格式、因子变量等

    return:
    --------
    var_type:dict{
        ColumnName:type,}

    '''
    assert isinstance(data,pd.core.frame.DataFrame)
    var_type={}
    for c in data.columns:
        #print('type_of_var : ',c)
        if copy:
            data=data.copy()
            result=dtype_detection(data[c],category_detection=category_detection,\
            criterion=criterion,min_mean_counts=min_mean_counts,datetime_to_category=False,fix=False)
            if result is not None:
                var_type[c]=result['vtype']
            else:
                var_type[c]='unknown'
        else:
            result,tmp=dtype_detection(data[c],category_detection=category_detection,\
            criterion=criterion,min_mean_counts=min_mean_counts,datetime_to_category=False,fix=True)
            data[c]=tmp
            if result is not None:
                var_type[c]=result['vtype']
            else:
                var_type[c]='unknown'
    return var_type



def var_detection(data,combine=True):
    '''检测整个数据的变量类型,内部使用，外部请用type_of_var
    parameter
    ---------
    data: 数据,DataFrame格式
    combine: 检测变量中是否有类似的变量，有的话则会合并。

    return
    ------
    var_list:[{'name':,'vtype':,'vlist':,'ordered':,'categories':,},]

    '''
    var_list=[]
    for c in data.columns:
        result,tmp=dtype_detection(data[c],fix=True)
        data[c]=tmp
        if result is not None:
            var_list.append(result)
    if not(combine):
        return var_list,data
    var_group=[]
    i=0
    pattern=re.compile(r'(.*?)(\d+)')
    while i < len(var_list)-1:
        v=var_list[i]
        vnext=var_list[i+1]
        if v['vtype']!='number' or vnext['vtype']!='number':
            i+=1
            continue
        tmp1=[]
        for vv in var_list[i:]:
            if vv['vtype']!='number':
                break
            w=re.findall(pattern,'%s'%vv['name'])
            if len(w)==0 or (len(w)>0 and len(w[0])<2):
                break
            tmp1.append((w[0][0],w[0][1]))
        if len(tmp1)<2:
            i+=1
            continue
        flag1=len(set([t[0] for t in tmp1]))==1
        flag2=np.diff([int(t[1]) for t in tmp1]).sum()==len(tmp1)-1
        if flag1 and flag2:
            var_group.append(list(range(i,i+len(tmp1))))
            i+=len(tmp1)
    var_group_new={}
    var_group_total=[]#将所有的分组ind加起来
    for vi in var_group:
        var_group_total+=vi
        name='{}-->{}'.format(var_list[vi[0]]['name'],var_list[vi[-1]]['name'])
        vlist=[var_list[v]['name'] for v in vi]
        vtype='group_number'
        tmp={'name':name,'vtype':vtype,'vlist':vlist,'ordered':True,'categories':vlist}
        var_group_new[vi[0]]=tmp
    var_list_new=[]
    var_list_have=[]
    for i,v in enumerate(var_list):
        if i not in var_group_total:
            v['vlist']=[v['name']]
            var_list_new.append(v)
            var_list_have+=v['vlist']
        elif i in var_group_total and v['name'] not in var_list_have:
            var_list_new.append(var_group_new[i])
            var_list_have+=var_group_new[i]['vlist']
    return var_list_new,data

def describe(data):
    '''
    对每个变量生成统计指标特征
    对于每一个变量，生成如下字段：
        数据类型：
        最大值/频数最大的那个： 
        最小值/频数最小的那个：
        均值/频数中间的那个：
        缺失率：
        范围/唯一数：
    '''

    data=pd.DataFrame(data)
    n_sample=len(data)
    var_type=type_of_var(data,copy=True)
    summary=pd.DataFrame(columns=data.columns,index=['dtype','max','min','mean','missing_pct','std/nuniue'])
    for c in data.columns:
        missing_pct=1-data[c].count()/n_sample
        if var_type[c] == 'number':
            max_value,min_value,mean_value=data[c].max(),data[c].min(),data[c].mean()
            std_value=data[c].std()
            summary.loc[:,c]=[var_type[c],max_value,min_value,mean_value,missing_pct,std_value]
        elif var_type[c] == 'category':
            tmp=data[c].value_counts()
            max_value,min_value=tmp.argmax(),tmp.argmin()
            mean_value_index=tmp[tmp==tmp.median()].index
            mean_value=mean_value_index[0] if len(mean_value_index)>0 else np.nan
            summary.loc[:,c]=[var_type[c],max_value,min_value,mean_value,missing_pct,len(tmp)]
        elif var_type[c] == 'datetime':
            max_value,min_value=data[c].max(),data[c].min()
            summary.loc[:,c]=[var_type[c],max_value,min_value,np.nan,missing_pct,np.nan]
        else:
            summary.loc[:,c]=[var_type[c],np.nan,np.nan,np.nan,missing_pct,np.nan]
    return summary



def plot(data,figure_type='auto',chart_type='auto',vertical=False,ax=None):
    '''auto choose the best chart type to draw the data
    paremeter
    -----------
    figure_type: 'mpl' or 'pptx' or 'html'
    chart_type: 'hist' or 'dist' or 'kde' or 'bar' ......

    return
    -------
    chart:dict format.
    .type: equal to figure_type
    .fig: only return if type == 'mpl'
    .ax:
    .chart_data:

    '''

    # 判别部分

    # 绘制部分
    data=pd.DataFrame(data)
    chart={}
    if figure_type in ['mpl','matplotlib']:
        chart['type']='mpl'
        if ax is None:
            fig,ax=plt.subplots()
        if chart_type in ['hist','kde']:
            for c in data.columns:
                sns.kdeplot(data[c].dropna(),shade=True,ax=ax)
            legend_label=ax.get_legend_handles_labels()
            if len(legend_label)>0 and len(legend_label[0])>0:
                ax.legend()
            ax.axis('auto')
        elif chart_type in ['dist']:
            for c in data.columns:
                sns.distplot(data[c].dropna(),ax=ax)
            legend_label=ax.get_legend_handles_labels()
            if len(legend_label)>0 and len(legend_label[0])>0:
                ax.legend()
            ax.axis('auto')
        elif chart_type in ['scatter']:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.axhline(y=0, linestyle='-', linewidth=1.2, alpha=0.6)
            ax.axvline(x=0, linestyle='-', linewidth=1.2, alpha=0.6)
            color=['blue','red','green','dark']
            if not isinstance(data,list):
                data=[data]
            for i,dd in enumerate(data):
                if '%s'%dd.iloc[:,0] != 'nan' or '%s'%dd.iloc[:,1] != 'nan':
                    ax.scatter(dd.iloc[:,0], dd.iloc[:,1], c=color[i], s=50,
                               label=dd.columns[1])
                    for _, row in dd.iterrows():
                        ax.annotate(row.name, (row.iloc[0], row.iloc[1]), color=color[i],fontproperties=myfont,fontsize=10)
            ax.axis('equal')
            legend_label=ax.get_legend_handles_labels()
            if len(legend_label)>0 and len(legend_label[0])>0:
                ax.legend()
        try:
            chart['fig']=fig
        except:
            pass
        chart['ax']=ax
        return chart



def AnalysisReport(data,filename=None,var_list=None):
    '''
    直接生成报告
    '''
    if var_list is None:
        var_list,data=var_detection(data)
        #print(var_list)
        #print('============')

    slides_data=[]

    if filename is None:
        filename='AnalysisReport'+time.strftime('_%Y%m%d%H%M', time.localtime())
        p=_rpt.Report()
        p.add_cover(title=os.path.splitext(filename)[0])
    elif isinstance(filename,str):
        p=_rpt.Report()
        p.add_cover(title=os.path.splitext(filename)[0])
    elif isinstance(filename,_rpt.Report):
        p=filename
        filename='AnalysisReport'+time.strftime('_%Y%m%d%H%M', time.localtime())
    else:
        print('reportgen.AnalysisReport::cannot understand the filename')
        return None

    summary=describe(data)
    n_cut=round(summary.shape[1]/10)
    n_cut=1 if n_cut==0 else n_cut
    for i in range(n_cut):
        if i!=n_cut-1:
            summary_tmp=summary.iloc[:,10*i:10*i+10]
        else:
            summary_tmp=summary.iloc[:,10*i:]
        slide_data={'data':summary_tmp,'slide_type':'table'}
        title='数据字段描述{}'.format(i+1) if n_cut>1 else '数据字段描述'
        p.add_slide(data=slide_data,title=title)

    for v in var_list:
        vtype=v['vtype']
        name=v['name']
        vlist=v['vlist']       
        #print(name,':',vtype)
        if vtype == 'number':
            chart=plot(data[name],figure_type='mpl',chart_type='kde')
            chart['fig'].savefig('kdeplot1.png',dpi=200)
            chart['fig'].clf()
            del chart
            chart=plot(data[name],figure_type='mpl',chart_type='dist')
            chart['fig'].savefig('kdeplot2.png',dpi=200)
            chart['fig'].clf()
            del chart
            summary='''平均数为：{:.2f}，标准差为：{:.2f}，最大为：{}'''\
            .format(data[name].mean(),data[name].std(),data[name].max())
            footnote='注: 样本N={}'.format(data[name].count())
            slide_data=[{'data':'kdeplot1.png','slide_type':'picture'},{'data':'kdeplot2.png','slide_type':'picture'}]
            p.add_slide(data=slide_data,title=name+' 的分析',summary=summary,footnote=footnote)
            slides_data.append(slide_data)
            os.remove('kdeplot1.png')
            os.remove('kdeplot2.png')
        elif vtype == 'category':
            tmp=pd.DataFrame(data[name].value_counts())
            tmp=tmp*100/tmp.sum()#转换成百分数
            if ('ordered' in v) and v['ordered']:
                tmp=pd.DataFrame(tmp,index=v['categories'])
            footnote='注: 样本N={}'.format(data[name].count())
            slide_data={'data':tmp,'slide_type':'chart','type':'COLUMN_CLUSTERED'}
            summary='{}占比最大为: {:.2f}%'.format(tmp.iloc[:,0].argmax(),tmp.iloc[:,0].max())
            p.add_slide(data=slide_data,title=name+' 的分析',summary=summary,footnote=footnote)
            slides_data.append(slide_data)
        elif vtype == 'datetime':
            if data[name].value_counts().max()==1:
                print('the dtype of {} column is datetime, continue...')
                continue
            tmp=pd.DataFrame(data[name].astype('object').value_counts())
            tmp=tmp*100/tmp.sum()#转换成百分数
            tmp=tmp.sort_index()#排序
            if ('ordered' in v) and v['ordered']:
                tmp=pd.DataFrame(tmp,index=v['categories'])
            footnote='注: 样本N={}'.format(data[name].count())
            slide_data={'data':tmp,'slide_type':'chart','type':'COLUMN_CLUSTERED'}
            summary='{}占比最大为: {:.2f}%'.format(tmp.iloc[:,0].argmax(),tmp.iloc[:,0].max())
            p.add_slide(data=slide_data,title=name+' 的分析',summary=summary,footnote=footnote)
            slides_data.append(slide_data)
        elif vtype == 'text':
            try:
                tmp=','.join(data[name].dropna())
                if len(tmp)>1:
                    img=genwordcloud(tmp,font_path=font_path)
                    img.save('tmp.png')
                    footnote='注: 样本N={}'.format(data[name].count())
                    slide_data={'data':'tmp.png','slide_type':'picture'}
                    p.add_slide(data=slide_data,title=name+' 的词云分析',footnote=footnote)
                    slides_data.append(slide_data)
                    os.remove('tmp.png')
            except:
                print('cannot understand : {}'.format(name))
                pass
        elif vtype == 'group_number':
            tmp=pd.DataFrame(data.loc[:,vlist].mean())
            footnote='注: 样本N={}'.format(data.loc[:,vlist].count().max())
            slide_data={'data':tmp,'slide_type':'chart','type':'COLUMN_CLUSTERED'}
            summary='{}占比最大为: {:.2f}%'.format(tmp.iloc[:,0].argmax(),tmp.iloc[:,0].max())
            p.add_slide(data=slide_data,title=name+' 的分析',summary=summary,footnote=footnote)
            slides_data.append(slide_data)
        elif vtype == 'text_st':
            print('The field: {} may be id or need to be designed'.format(name))
        else:
            print('unknown type: {}'.format(name))
    p.save(os.path.splitext(filename)[0]+'.pptx')
    return slides_data
    
    
    
def ClassifierReport(y_true,y_preds,y_probas,img_save=False):
    '''二分类模型评估（后期可能会修改为多分类）
    真实数据和预测数据之间的各种可视化和度量
    
    parameters:
    -----------
    y_true: array_like 真实的标签,binary
    y_preds: dict or array_like. 预测的标签，binary,可以用 dict 存储多个模型的预测标签数据
    y_probas: dict or array_like. 预测的概率，0-1,可以用 dict 存储多个模型的预测标签数据
    img_save：Bool，是否直接将图片保存到本地
    
    return:
    ---------
    models_report: 各模型的各种评估数据
    conf_matrix: 各模型的混淆矩阵
    '''


    #from sklearn import metrics
    assert type(y_preds) == type(y_probas)
    if not(isinstance(y_preds,dict)):
        y_preds={'clf':y_preds}
        y_probas={'clf':y_probas}
    models_report=pd.DataFrame()
    conf_matrix={}
    fig1,ax1=plt.subplots()
    fig2,ax2=plt.subplots()
    fig3,ax3=plt.subplots()
    for clf in y_preds:
        y_pred=y_preds[clf]
        y_proba=y_probas[clf]
        try:
            kl_div_score=entropyc.kl_div(y_proba[y_true==1],y_proba[y_true==0])
            kl_div_score+=entropyc.kl_div(y_proba[y_true==0],y_proba[y_true==1])
        except:
            kl_div_score=np.nan
        scores = pd.Series({'model' : clf,
                            'roc_auc_score' : metrics.roc_auc_score(y_true, y_proba),
                             'good_rate': y_true.value_counts()[0]/len(y_true),
                             'matthews_corrcoef': metrics.matthews_corrcoef(y_true, y_pred),
                             'accuracy_score': metrics.accuracy_score(y_true,y_pred),
                             'ks_score': np.nan,
                             'precision_score': metrics.precision_score(y_true, y_pred),
                             'recall_score': metrics.recall_score(y_true, y_pred),
                             'kl_div': kl_div_score,
                             'f1_score': metrics.f1_score(y_true, y_pred)})
        models_report=models_report.append(scores,ignore_index = True)
        conf_matrix[clf]=pd.crosstab(y_true, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
        #print('\n{} 模型的混淆矩阵:'.format(clf))
        #print(conf_matrix[clf])

        # ROC 曲线
        fpr, tpr, thresholds=metrics.roc_curve(y_true,y_proba,pos_label=1)
        auc_score=metrics.auc(fpr,tpr)
        w=tpr-fpr
        ks_score=w.max()
        models_report.loc[models_report['model']==clf,'ks_score']=ks_score
        ks_x=fpr[w.argmax()]
        ks_y=tpr[w.argmax()]
        #sc=thresholds[w.argmax()]
        #fig1,ax1=plt.subplots()
        ax1.set_title('ROC Curve')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
        ax1.plot([ks_x,ks_x], [ks_x,ks_y], '--', color='red')
        ax1.text(ks_x,(ks_x+ks_y)/2,r'   $S_c$=%.2f, KS=%.3f'%(thresholds[w.argmax()],ks_score))
        ax1.plot(fpr,tpr,label='{}:AUC={:.5f}'.format(clf,auc_score))
        ax1.legend()
        # PR 曲线
        precision, recall, thresholds=metrics.precision_recall_curve(y_true,y_proba,pos_label=1)
        #fig2,ax2=plt.subplots()
        ax2.plot(recall,precision,label=clf)
        ax2.set_title('P-R Curve')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend()
        #fig2.show()
        #密度函数和KL距离
        #fig3,ax3=plt.subplots()
        sns.kdeplot(y_proba[y_true==0],ax=ax3,shade=True,label='{}-0'.format(clf))
        sns.kdeplot(y_proba[y_true==1],ax=ax3,shade=True,label='{}-1'.format(clf))
        ax3.set_title('Density Curve')
        ax3.legend()
        ax3.autoscale()
        #fig3.show()


    if img_save:
        fig1.savefig('roc_curve_{}.png'.format(time.strftime('%Y%m%d%H%M', time.localtime())),dpi=400)
        fig2.savefig('pr_curve_{}.png'.format(time.strftime('%Y%m%d%H%M', time.localtime())),dpi=400)
        fig3.savefig('density_curve_{}.png'.format(time.strftime('%Y%m%d%H%M', time.localtime())),dpi=400)
    else:
        fig1.show()
        fig2.show()
        fig3.show()
    models_report=models_report.set_index('model')
    #print('模型的性能评估:')
    #print(models_report)
    return models_report,conf_matrix
    
    
    
    
    
