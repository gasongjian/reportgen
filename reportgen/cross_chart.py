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


def contingency(df,col_dis=None,row_dis=None,alpha=0.05):
    '''
    列联表分析：
    1、生成三种百分表
    2、如果存在两个变量，则进行列联分析
    3、当变量存在多个时，进一步进行两两独立性检验
    需要注意的问题：
    1、题目分为单选(互斥变量)和多选(非互斥变量)，非互斥变量无形中增加了总样本量，默认把需
    要检验的变量放在列中，且要求是互斥变量,如若行是非互斥变量，则需要提供互斥变量的样本频数表

    返回字典结果体
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



# =================[问卷网数据格式化]==========================
d2,code=rpt.wenjuanwang(filepath='.\\questionnaire_data')


# ================数据清洗=================
d2=d2[d2['Q49'].notnull()]
d2['Q49'].replace({5:u'时尚',6:u'商务',7:u'科技',8:u'跟随'},inplace=True)
code['Q49']['code']={5:u'时尚',6:u'商务',7:u'科技',8:u'跟随'}


# ==============交叉分析题设置==============
cross_class='Q49'#交叉分析的类别
delclass=None#只允许一类，便于统计用
filename=u'人群差异'
# ==============交叉分析题设置==============



prs = Presentation()
prs1 = Presentation()# 用于统计TGI指数

# ======================相关参数
max_column_chart=20

#total_qlist=list(code.keys())
total_qlist_0=['Q'+'%s'%(i+1) for i in range(len(code.keys()))]


# ===========统计具有显著性差异的数字和题目
difference=dict(zip(total_qlist_0,[-1]*len(total_qlist_0)))
difference[cross_class]=-2 #-2就代表了是交叉题目

# 需要从列表中删除的题目
remove_list=['Q1',cross_class]
total_qlist=[x for x in total_qlist_0 if x not in remove_list]
# 交叉变量类别列表
cross_class_list=list(d2[cross_class].unique())
if delclass:
    cross_class_list.remove(delclass)

# 每个类别的样本数
cn=d2[cross_class].value_counts()
cn[u'总体']=cn.sum()


# ================背景页(后期直接在report包中封装)
title=u'背景说明'
summary=u'交叉题目为'+cross_class+u': '+code[cross_class]['content']
summary=summary+'\n'+u'各类别样本量如下：'
rpt.plot_table(prs,cn,title=title,summary=summary)
rpt.plot_table(prs1,cn,title=title,summary=summary)



for qq in total_qlist:
    '''
    特殊题型处理
    整体满意度题：后期归为数值类题型
    '''

    qtitle=code[qq]['content']
    qlist=code[qq]['qlist']
    if (u'满意度' in qtitle) and (u'整体' in qtitle):
        qtype2=u'整体满意度'
        t=d2.groupby([cross_class])[[qq]].mean()-1
        chi2_data=chi2_test(pd.crosstab(d2[qq],d2[cross_class]))
        title=u'整体满意度'
        if chi2_data[0]:
            difference[qq]=1
            summary=u'卡方检验 p值为 %s <0.05, 即存在显著性差异' %(chi2_data[1])
        else:
            difference[qq]=0
            summary=u'卡方检验 p值为 %.2f >0.05, 即不存在显著性差异' %(chi2_data[1])
        rpt.plot_chart(prs,t,'COLUMN_CLUSTERED',title=title,summary=summary)
        rpt.plot_chart(prs1,t,'COLUMN_CLUSTERED',title=title,summary=summary)
        continue

    if (u'意愿' in qtitle) and (u'推荐' in qtitle):
        qtype2='NPS'
        t=pd.crosstab(d2[qq],d2[cross_class])
        chi2_data=chi2_test(t)
        t[u'总体']=d2[qq].value_counts()
        for i in range(len(t.columns)):
            t.iloc[:,i]=t.iloc[:,i]/cn[t.columns[i]]
        NPS=t.iloc[9:11,:].sum()-t.iloc[0:7,:].sum()
        NPS=pd.DataFrame(NPS)
        NPS.rename(columns={0:'NPS'},inplace=True)
        title=u'NPS'
        if chi2_data[0]:
            difference[qq]=1
            summary=u'卡方检验 p值为 %s <0.05, 即存在显著性差异' %(chi2_data[1])
        else:
            difference[qq]=0
            summary=u'卡方检验 p值为 %.2f >0.05, 即不存在显著性差异' %(chi2_data[1])
        rpt.plot_chart(prs,NPS,'COLUMN_CLUSTERED',title=title,summary=summary)
        rpt.plot_chart(prs1,NPS,'COLUMN_CLUSTERED',title=title,summary=summary)
        continue
    '''
    if (u'城市' in qtitle) and (code[qq]['qtype'] == u'下拉填空题'):
        qtype2=u'城市'
        # 省份
        t=pd.crosstab(d2[qlist[0]],d2[cross_class])
        title=qq+': '+code[qq]['content']+u'_省份'
        summary=u'不满足卡方检验条件们，暂时不提供显著性检验.'
        if len(t)>max_column_chart:
            rpt.plot_chart(prs,t,'BAR_CLUSTERED',title=title,summary=summary)
            rpt.plot_chart(prs1,t,'BAR_CLUSTERED',title=title,summary=summary)
        else:
            rpt.plot_chart(prs,t,'COLUMN_CLUSTERED',title=title,summary=summary)
            rpt.plot_chart(prs1,t,'COLUMN_CLUSTERED',title=title,summary=summary)
        # 城市
        t=pd.crosstab(d2[qlist[1]],d2[cross_class])
        title=qq+': '+code[qq]['content']+u'_城市'
        summary=u'不满足卡方检验条件们，暂时不提供显著性检验.'
        if len(t)>max_column_chart:
            rpt.plot_chart(prs,t,'BAR_CLUSTERED',title=title,summary=summary)
            rpt.plot_chart(prs1,t,'BAR_CLUSTERED',title=title,summary=summary)
        else:
            rpt.plot_chart(prs,t,'COLUMN_CLUSTERED',title=title,summary=summary)
            rpt.plot_chart(prs1,t,'COLUMN_CLUSTERED',title=title,summary=summary)
        continue
    '''


    if code[qq]['qtype'] in [u'单选题',u'多选题']:
        if code[qq]['qtype'] == u'单选题':
            t=pd.crosstab(d2[qq],d2[cross_class])
            t[u'总体']=d2[qq].value_counts()
        else:
            qlist=code[qq]['qlist']
            if re.findall('_open',code[qq]['qlist'][-1]):
                qlist=qlist[:-1]
            t=d2.groupby([cross_class])[qlist].sum().T
            t[u'总体']=d2[qlist].sum()
        t.rename(index=code[qq]['code'],inplace=True)
        # 小样本剔除
        #t=t[t[u'总体']>=30] #剔除样本量小于30的行指标
        if u'其他【请注明】' in t.index:
            t.drop([u'其他【请注明】'],axis=0,inplace=True)
        if delclass:
            t.drop([delclass],axis=1,inplace=True)
        # 卡方检验
        chi2_data=chi2_test(t.drop([u'总体'],axis=1))
        # 把样本频数改成百分比
        for i in range(len(t.columns)):
            t.iloc[:,i]=t.iloc[:,i]/cn[t.columns[i]]
        # 生成TGI指数
        t_TGI=t.copy()
        for i in range(len(t.columns)):
            t_TGI.iloc[:,i]=t_TGI.iloc[:,i]/t_TGI[u'总体']
        # 剔除t中的总体
        t_TGI.drop([u'总体'],axis=1,inplace=True)
        t.drop([u'总体'],axis=1,inplace=True)
        #t.to_csv(qq+'.csv',encoding='gbk')
        title=qq+': '+code[qq]['content']
        if chi2_data[0]:
            difference[qq]=1
            summary=u'卡方检验 p值为 %s <0.05, 即存在显著性差异' %(chi2_data[1])
        else:
            difference[qq]=0
            summary=u'卡方检验 p值为 %.2f >0.05, 即不存在显著性差异' %(chi2_data[1])
        #summary=summary+'\n'+u'我是结论'
        if len(t)>max_column_chart:
            rpt.plot_chart(prs,t,'BAR_CLUSTERED',title=title,summary=summary)
            rpt.plot_chart(prs1,t_TGI,'BAR_CLUSTERED',title=title,summary=summary)
        else:
            rpt.plot_chart(prs,t,'COLUMN_CLUSTERED',title=title,summary=summary)
            rpt.plot_chart(prs1,t_TGI,'COLUMN_CLUSTERED',title=title,summary=summary)
        continue

    if code[qq]['qtype'] in [u'排序题']:
        qlist=code[qq]['qlist']
        for c in cross_class_list:
            dd=d2[d2[cross_class]==c]
            t=pd.DataFrame()
            for q in qlist:
                tmp=dd[q].value_counts()
                t=pd.concat([t,tmp],axis=1)
            t.rename(index={1:'TOP1',2:'TOP2',3:'TOP3',4:'TOP4',5:'TOP5'},inplace=True)
            t.rename(columns=code[qq]['code'],inplace=True)
            t.replace({np.nan:0},inplace=True)
            t=t.T
            t.sort_values(by=['TOP1'],ascending=0,inplace=True)
            title=qq+': '+u'排序题__'+c
            summary=u'不满足卡方检验条件们，暂时不提供显著性检验.'
            if len(t)>max_column_chart:
                rpt.plot_chart(prs,t,'BAR_STACKED',title=title,summary=summary)
                rpt.plot_chart(prs1,t,'BAR_STACKED',title=title,summary=summary)
            else:
                rpt.plot_chart(prs,t,'COLUMN_STACKED',title=title,summary=summary)
                rpt.plot_chart(prs1,t,'COLUMN_STACKED',title=title,summary=summary)
            continue

    if code[qq]['qtype'] in [u'矩阵单选题']:
        #此处将顺序因子变量处理成连续数值变量，可能会有些不妥
        qlist=code[qq]['qlist']
        t=d2.groupby([cross_class])[qlist].mean().T
        t.rename(index=code[qq]['code_r'],inplace=True)
        title=qq+': '+code[qq]['content']
        summary=u'不满足卡方检验条件们，暂时不提供显著性检验.'
        if len(t)>max_column_chart:
            rpt.plot_chart(prs,t,'BAR_CLUSTERED',title=title,summary=summary)
            rpt.plot_chart(prs1,t,'BAR_CLUSTERED',title=title,summary=summary)
        else:
            rpt.plot_chart(prs,t,'COLUMN_CLUSTERED',title=title,summary=summary)
            rpt.plot_chart(prs1,t,'COLUMN_CLUSTERED',title=title,summary=summary)
        continue


# ==============小结页
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

# ========================文件生成和导出
difference.name=cross_class
difference=pd.DataFrame(difference)
if not os.path.exists('.\\out'):
    os.mkdir('.\\out')
difference.to_csv('.\\out\\'+filename+u'_显著性检验.csv',encoding='gbk')
prs.save('.\\out\\'+filename+u'_百分比.pptx')
prs1.save('.\\out\\'+filename+u'_TGI指数.pptx')