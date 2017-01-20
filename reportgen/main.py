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

# =================[后期会封装成单独的函数到report包中]=========================
'''
后期添加功能：
1、结论自动提取
'''


def chi2_test(df,alpha=0.5):
    import scipy.stats as stats
    df=pd.DataFrame(df)
    chiStats = stats.chi2_contingency(observed=df)
    alpha = 0.05
    #critical_value = stats.chi2.ppf(q=1-alpha,df=chiStats[2])
    #observed_chi_val = chiStats[0]
    # p<alpha 等价于 observed_chi_val>critical_value
    chi2_data=(chiStats[1] <= alpha,chiStats[1])
    return chi2_data

def rpttable(data,code):
    '''
    单个题目描述统计
    code是data的编码，列数大于1
    返回两个数据：
    t1：默认的百分比表
    t2：原始频数表，且添加了合计项
    '''
    # 单选题
    qtype=code['qtype']
    index=code['qlist']
    if qtype == u'单选题':
        t1=data.value_counts()
        t=t1/data.notnull().sum()*1.0           
    elif qtype == u'多选题':
        t1=data.sum()
        t=t1/data.iloc[:,0].notnull().sum()*1.0
    elif qtype == u'矩阵单选题':
        t1=data.mean()
        t=t1.copy()
    elif qtype == u'排序题':
        topn=max([len(data[q][data[q].notnull()].unique()) for q in index])
        qsort=dict(zip([i+1 for i in range(topn)],[topn-i for i in range(topn)]))
        data.replace(qsort,inplace=True)
        t1=data.sum()
        t=t1/data.iloc[:,0].notnull().sum()*1.0
    t.rename(index=code['code'],inplace=True)
    t1.rename(index=code['code'],inplace=True)
    return (t,t1)




def rptcrosstab(data_index,data_column,code):
    '''
    默认data_column=data[cross_class] 是交叉分析列标签对应的数据，列数==1
    code是data_index 的编码，列数大于1
    返回两个数据：
    t1：默认的百分比表
    t2：原始频数表，且添加了合计项
    '''
    # 单选题
    data_index=pd.DataFrame(data_index)
    data_column=pd.DataFrame(data_column)
    qtype=code['qtype']
    cross_class=data_column.columns[0]
    #columns=data_column[data_column.notnull()].unique()
    index=code['qlist']
    #len_index=d2[current_name].notnull().sum()
    #len_column=data_column.notnull().sum()
    column_freq=data_column.iloc[:,0].value_counts()
    column_freq[u'总体']=column_freq.sum()
    #index_freq=data[cross_class].value_counts()
    if qtype == u'单选题':
        t=pd.crosstab(data_index.iloc[:,0],data_column.iloc[:,0])
        t[u'总体']=data_index.iloc[:,0].value_counts()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        for i in t.columns:
            t.loc[:,i]=t.loc[:,i]/t.sum()[i]
        #t1.loc[u'合计',:]=column_freq
        t.rename(index=code['code'],inplace=True)
        t1.rename(index=code['code'],inplace=True)
        return (t,t1)
            
    elif qtype == u'多选题':
        data=data_index.join(data_column)
        t=data.groupby([cross_class])[index].sum().T
        t[u'总体']=data_index.sum()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        for i in t.columns:
            t.loc[:,i]=t.loc[:,i]/column_freq[i]
        #t1.loc[u'合计',:]=column_freq
        t.rename(index=code['code'],inplace=True)
        t1.rename(index=code['code'],inplace=True)
        return (t,t1)
    elif qtype == u'矩阵单选题':
        data=data_index.join(data_column)
        t=data.groupby([cross_class])[index].mean().T
        t[u'总体']=data_index.mean()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        #t1.loc[u'合计',:]=column_freq
        t.rename(index=code['code_r'],inplace=True)
        t1.rename(index=code['code_r'],inplace=True)
        return (t,t1)
    elif qtype == u'排序题':
        data=data_index.join(data_column)
        topn=max([len(data_index[q][data_index[q].notnull()].unique()) for q in index])
        qsort=dict(zip([i+1 for i in range(topn)],[topn-i for i in range(topn)]))
        data_index.replace(qsort,inplace=True)
        t=data.groupby([cross_class])[index].sum().T
        t[u'总体']=data_index.sum()/data_index.iloc[:,0].notnull().sum()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        for i in t.columns:
            t.loc[:,i]=t.loc[:,i]/column_freq[i]
        #t1.loc[u'合计',:]=column_freq
        t.rename(index=code['code'],inplace=True)
        t1.rename(index=code['code'],inplace=True)
        return (t,t1)
        '''
        t=pd.DataFrame(columns=columns,index=index)
        for i in t.index:
            for j in t.columns:
                tmp=data_index.loc[data_column==j,i]
                tmp=tmp*(topn+1-tmp)
                t.loc[i,j]=tmp.sum()
        '''
    else:
        return (None,None)





def contingency(df,col_dis=None,row_dis=None,alpha=0.05):
    '''
    列联表分析：
    1、生成三种百分表
    2、如果存在两个变量，则进行列联分析
    3、当变量存在多个时，进一步进行两两独立性检验
    需要注意的问题：
    1、题目分为单选(互斥变量)和多选(非互斥变量)，非互斥变量无形中增加了总样本量，默认把需
    要检验的变量放在列中，且要求是互斥变量,如若行是非互斥变量，则需要提供互斥变量的样本频数表
    默认行是qq，列是cross_class
    输入：
    col_dis: 列变量的频数分布
    row_dis: 行变量的频数分布
    返回字典格式
    chi_test: 卡方检验结果，1:显著；0:不显著；-1：期望值不满足条件
    coef: 包含chi2、p值、V相关系数
    log: 记录一些异常情况
    significant:显著性结果
    percentum:百分比表，包含_col,_row,_total
    '''
    import scipy.stats as stats
    conti_data={}
    if isinstance(df,pd.core.series.Series):
        df=pd.DataFrame(df)
    R,C=df.shape
    if not col_dis:
        col_dis=df.sum()
    if not row_dis:
        row_dis=df.sum(axis=1)
    sample=min(col_dis.sum(),row_dis.sum())#样本总数
    n=df.sum().sum()# 列联表中所有频数之和
    conti_len=R*C# 列联表的单元数

    # 卡方检验
    threshold=ceil(conti_len*0.2)# 期望频数和实际频数不得小于5
    chiStats = stats.chi2_contingency(observed=df)
    fe=chiStats[3]# 期望频数
    if chiStats[1] <= alpha:
        conti_data['chi_test']=1
    else:
        conti_data['chi_test']=0

    chi_data=chiStats[0]
    p_value=chiStats[1]
    vcoef=sqrt(chi_data/n/min(R-1,C-1))#V相关系数，用于显著性比较
    conti_data['coef']={'p_value':p_value,'chi_data':chi_data,'vcoef':vcoef}
    if sum(sum(fe<=5))>=threshold:
        conti_data['chi_test']=-1
        conti_data['log']=u'期望频数小于5的单元数过多'

    chi_sum=(df.as_matrix()-fe)**2/fe
    chi_sum=chi_sum.sum(axis=1)
    chi_value_fit=stats.chi2.ppf(q=1-alpha,df=C-1)#拟合优度检验
    significant=list(df.index[np.where(chi_sum>chi_value_fit)])
    conti_data['significant']=significant

    # 把样本频数改成百分比
    df[u'合计']=row_dis
    for i in range(C):
        df.iloc[:,i]=df.iloc[:,i]/col_dis[df.columns[i]]
    # 生成TGI指数
    df_TGI=df.copy()
    for i in range(C):
        df_TGI.iloc[:,i]=df_TGI.iloc[:,i]/df_TGI[u'合计']
    # 剔除t中的总体
    df_TGI.drop([u'合计'],axis=1,inplace=True)
    df.drop([u'合计'],axis=1,inplace=True)


'''
data 和 code必须有，且code必须有以下变量：
code.qlist
code.content
code.qtype
'''

# =================[数据导入]==========================
d,code=rpt.wenjuanxing()


#data=d[['Q37','Q35','Q36','Q39','Q8']+code['Q10']['qlist']+code['Q12']['qlist']]
data=d.copy()
data['Q37'].replace(code['Q37']['code'],inplace=True)


def cross_chart(data,cross_class, filename=u'交叉分析', cross_qlist=None,\
delclass=None,data_style=None,cross_order=None, significance_test=False, \
total_display=True,max_column_chart=20):


    # ===================参数预处理=======================
    data_style=data_style.upper()
    if significance_test:
        difference=dict(zip(total_qlist_0,[-1]*len(total_qlist_0)))
        difference[cross_class]=-2 #-2就代表了是交叉题目
    
    if not cross_qlist:
        try:
            cross_qlist=list(sorted(code,key=lambda c:int(c[1:])))
        except:
            cross_qlist=list(code.keys())
        cross_qlist.remove(cross_class)
    
    # =================基本数据获取==========================
    
    # 交叉变量中每个类别的频数分布
    cross_class_freq=data[cross_class].value_counts()
    cross_class_freq[u'总体']=cross_class_freq.sum()
    # 前期统一替换选项编码
    data[cross_class].replace(code[cross_class]['code'])
    #交叉分析的样本数统一为交叉变量的样本数
    sample_len=data[cross_class].notnull().sum()
    cn=data[cross_class].value_counts()
    cn[u'合计']=cn.sum()
    
    # ================I/O接口=============================
    prs = Presentation()
    if not os.path.exists('.\\out'):
        os.mkdir('.\\out')
    Writer=pd.ExcelWriter('.\\out\\'+filename+'.xlsx')
    # ================背景页=============================
    title=u'背景说明(Powered by Python)'
    summary=u'交叉题目为'+cross_class+u': '+code[cross_class]['content']
    summary=summary+'\n'+u'各类别样本量如下：'
    rpt.plot_table(prs,cn,title=title,summary=summary)
    
    
    for qq in cross_qlist:
        '''
        特殊题型处理
        整体满意度题：后期归为数值类题型
        '''
    
        qtitle=code[qq]['content']
        qlist=code[qq]['qlist']
        qtype=code[qq]['qtype']
        if qtype not in [u'单选题',u'多选题',u'排序题',u'矩阵单选题']:
            continue
        t,t1=rptcrosstab(data[qlist],data[[cross_class]],code=code[qq])
        
        # =======数据修正==============
        if cross_order:
            cross_order=[q for q in cross_order if q in t.columns]
            t=pd.DataFrame(t,columns=cross_order)
            t1=pd.DataFrame(t1,columns=cross_order)
        if 'code_order' in code[qq]:
            t=pd.DataFrame(t,index=code_order)
            t1=pd.DataFrame(t1,index=code_order)
        t.fillna(0,inplace=True)
        t1.fillna(0,inplace=True)
        t2=pd.concat([t,t1],axis=1)
        
        # =======保存到Excel中========
        t2.to_excel(Writer,qq)
        
        '''列联表分析[暂缺]
        cc=contingency(t,col_dis=None,row_dis=None,alpha=0.05)
        
        '''
    
    
    
        '''
        # ========================【特殊题型处理区】================================
        if ('name' in code[qq].keys()) and code[qq]['name'] in [u'满意度','satisfaction']:
            title=u'整体满意度'
            
            
            
        # ========================【特殊题型处理区】================================
        '''
    
        title=qq+': '+qtitle
        summary=u'这里是结论区域.'
        footnote=u'样本N=%d'%sample_len
        if (not total_display) and (u'总体' in t.columns):
            t.drop([u'总体'],axis=1,inplace=True)
        if len(t)>max_column_chart:
            rpt.plot_chart(prs,t,'BAR_CLUSTERED',title=title,summary=summary,footnote=footnote)
        else:
            rpt.plot_chart(prs,t,'COLUMN_CLUSTERED',title=title,summary=summary,footnote=footnote)
    
    
    
    
    '''
    # ==============小结页=====================
    difference=pd.Series(difference,index=total_qlist_0)
    n1=sum(difference==1)
    n2=sum(difference==0)
    chi2_test_yes=list(difference[difference==1].index)
    title=u'小结'
    summary=u'一共卡方检验了 %s 道题, 其中有 %s 道题呈现显著性差异.'%(n1+n2,n1)
    summary=summary+'\n'+u'有显著性差异的分别是：'
    summary=summary+'\n'+', '.join(chi2_test_yes)
    summary=summary+'\n'+u'具体细节可查看文件: _显著性检验.csv'
    
    #rpt.plot_textbox(prs,title=title,summary=summary)
    #rpt.plot_textbox(prs1,title=title,summary=summary)
    '''
    
    # ========================文件生成和导出======================
    #difference.to_csv('.\\out\\'+filename+u'_显著性检验.csv',encoding='gbk')
    prs.save('.\\out\\'+filename+u'.pptx')
    Writer.save()



#def cross_chart(data,cross_class, filename=u'交叉分析', cross_qlist=None,\
#delclass=None,data_style=None,cross_order=None, significance_test=False, \
#total_display=True,max_column_chart=20):

data
filename=u'描述统计报告'
significance_test=False
summary_qlist=None




# ===================参数预处理=======================
if significance_test:
    difference=dict(zip(total_qlist_0,[-1]*len(total_qlist_0)))
    difference[cross_class]=-2 #-2就代表了是交叉题目
if not summary_qlist:
    try:
        summary_qlist=list(sorted(code,key=lambda c:int(c[1:])))
    except:
        summary_qlist=list(code.keys())


# =================基本数据获取==========================
#统一的有效样本，各个题目可能有不能的样本数
sample_len=len(d)

# ================I/O接口=============================
prs = Presentation()
if not os.path.exists('.\\out'):
    os.mkdir('.\\out')
Writer=pd.ExcelWriter('.\\out\\'+filename+'.xlsx')
# ================背景页=============================
title=u'背景说明(Powered by Python)'
summary=u'有效样本为%d'%sample_len
rpt.plot_textbox(prs,title=title,summary=summary)


for qq in cross_qlist:
    '''
    特殊题型处理
    整体满意度题：后期归为数值类题型
    '''

    qtitle=code[qq]['content']
    qlist=code[qq]['qlist']
    qtype=code[qq]['qtype']
    sample_len_qq=data[qlist[0]].notnull().sum()
    if qtype not in [u'单选题',u'多选题',u'排序题',u'矩阵单选题']:
        continue
    t=rptcrosstab(data[qlist],code=code[qq])
    
    # =======数据修正==============
    if 'code_order' in code[qq]:
        t=pd.DataFrame(t,index=code_order)
        t1=pd.DataFrame(t1,index=code_order)
    t.fillna(0,inplace=True)    
    # =======保存到Excel中========
    t.to_excel(Writer,qq)
    
    '''显著性分析[暂缺]
    cc=contingency(t,col_dis=None,row_dis=None,alpha=0.05)
    
    '''



    '''
    # ========================【特殊题型处理区】================================
    if ('name' in code[qq].keys()) and code[qq]['name'] in [u'满意度','satisfaction']:
        title=u'整体满意度'
        
        
        

    '''

    title=qq+': '+qtitle
    summary=u'这里是结论区域.'
    footnote=u'样本N=%d'%sample_len_qq
    if len(t)>max_column_chart:
        rpt.plot_chart(prs,t,'BAR_CLUSTERED',title=title,summary=summary,footnote=footnote)
    else:
        rpt.plot_chart(prs,t,'COLUMN_CLUSTERED',title=title,summary=summary,footnote=footnote)




'''
# ==============小结页=====================
difference=pd.Series(difference,index=total_qlist_0)
n1=sum(difference==1)
n2=sum(difference==0)
chi2_test_yes=list(difference[difference==1].index)
title=u'小结'
summary=u'一共卡方检验了 %s 道题, 其中有 %s 道题呈现显著性差异.'%(n1+n2,n1)
summary=summary+'\n'+u'有显著性差异的分别是：'
summary=summary+'\n'+', '.join(chi2_test_yes)
summary=summary+'\n'+u'具体细节可查看文件: _显著性检验.csv'

#rpt.plot_textbox(prs,title=title,summary=summary)
#rpt.plot_textbox(prs1,title=title,summary=summary)
'''

# ========================文件生成和导出======================
#difference.to_csv('.\\out\\'+filename+u'_显著性检验.csv',encoding='gbk')
prs.save('.\\out\\'+filename+u'.pptx')
Writer.save()
