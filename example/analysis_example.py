# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 17:57:48 2018

@author: gason
"""
import pandas as pd
import numpy as np
import reportgen as rpt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV

import warnings
warnings.filterwarnings('ignore') #为了整洁，去除弹出的warnings
pd.set_option('precision', 5) #设置精度
pd.set_option('display.float_format', lambda x: '%.5f' % x) #为了直观的显示数字，不采用科学计数法
pd.options.display.max_rows = 200 #最多显示200行




# 数据导入
data=pd.read_excel('.\\datasets\\LendingClub_Sample.xlsx')

# 数据预览
rpt.AnalysisReport(data.copy(),filename='LendingClub 数据预览');

# 机器学习相关函数补充

# 只作工具包测试，所以不区分训练集和测试集
y=data['target']
X=data.drop(['target'],axis=1)


# convert into dummies
categorical_var=list(set(X.columns[X.apply(pd.Series.nunique)<30])|set(X.select_dtypes(include=['O']).columns))
#categorical_var = ['collections_12_mths_ex_med', 'home_ownership', 'sub_grade',\
#'inq_last_6mths', 'initial_list_status', 'emp_length', 'application_type', \
#'acc_now_delinq', 'grade', 'purpose', 'verification_status', 'addr_state', 'term', 'pub_rec', 'delinq_2yrs']

continuous_var=list(set(X.columns)-set(categorical_var))
#continuous_var=['open_acc', 'total_rev_hi_lim', 'loan_amnt', 'tot_coll_amt', \
#'total_acc', 'tot_cur_bal', 'dti', 'annual_inc', 'earliest_cr_line', 'int_rate', 'installment']

# WOE 编码
woe=rpt.preprocessing.WeightOfEvidence(categorical_features=categorical_var,encoder_na=False)
X=woe.fit_transform(X,y)

# 离散化
#dis=rpt.preprocessing.Discretization(continous_features=continuous_var)
#X2=dis.fit_transform(X,y)

# 补缺和标准化
X=X.fillna(-99)
X[continuous_var]=preprocessing.MinMaxScaler().fit_transform(X[continuous_var])


clfs={'LogisticRegression':LogisticRegressionCV(),\
'RandomForest':RandomForestClassifier(),'GradientBoosting':GradientBoostingClassifier()}
y_preds,y_probas={},{}
for clf in clfs:
    clfs[clf].fit(X, y)
    y_preds[clf] =clfs[clf].predict(X)
    y_probas[clf] = clfs[clf].predict_proba(X)[:,1]

models_report,conf_matrix=rpt.ClassifierReport(y,y_preds,y_probas)
print(models_report)


# 信息论度量
p=y_probas['LogisticRegression'][y==1]
q=y_probas['LogisticRegression'][y==0]
print(rpt.metrics.entropyc.kl_div(p,q))


def xiu(data):
    data.iloc[:,0]=1
    return 2










