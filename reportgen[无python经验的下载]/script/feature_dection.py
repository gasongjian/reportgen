# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import report as rpt
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


index_X=list(X.index)
columns_X=list(X.columns)
index_Y=list(Y.index)
X=X.as_matrix()
Y=pd.Series(Y.as_matrix())


modelList=['逻辑回归','随机森林','决策树','SVM','线性回归']
model={}
for modelName in modelList:
    if modelName in ['逻辑回归']:
        model.update({'LogisticRegression':LogisticRegression()})
    elif modelName in ['随机森林']:
        model.update({'RandomForestRegressor':RandomForestRegressor()})
    elif modelName in ['决策树']:
        model.update({'DecisionTreeClassifier':DecisionTreeClassifier()})
    elif modelName in ['SVM']:
        model.update({'SVC':SVC()})
    elif modelName in ['线性回归']:
        model.update({'LinearRegression':LinearRegression()})

model1 = LogisticRegression()
model2 = RandomForestRegressor()
model3 = SVC()
model4 = LinearRegression()
model5 = DecisionTreeClassifier()

model_result={}
'''
'predice':预测值
'confusion_matrix':混淆矩阵
'精确率':
各特征的权重
'''

for clf in model:
    model[clf]=model[clf].fit(X,Y)
    Y_predict=np.round(model[clf].predict(X))
    Y_predict=pd.Series(Y_predict)
    confusion_matrix=pd.crosstab(Y,Y_predict)
    rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率

import pydotplus
from sklearn import tree
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("NPS.pdf")

model=model4
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print(rate1)
print(confusion_matrix)
weight=model.coef_
weight=pd.DataFrame(weight,index=qlist,columns=[u'线性回归'])
tmp=(weight-weight.min()+0.01)/(weight.max()-weight.min())
tmp=tmp/tmp.sum()
tmp.rename(columns={u'线性回归':u'线性回归标准化'},inplace=True)
weight=weight.join(tmp)


model=model2
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
#weight=model.feature_importances_
print(rate1)
print(confusion_matrix)
tmp=model.feature_importances_
tmp=pd.DataFrame(tmp,index=qlist,columns=[u'随机森林'])
weight=weight.join(tmp)




'''
model=model3
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print(rate1)
print(confusion_matrix)

model=model1
model = model.fit(X,Y)
Y_predict=np.round(model.predict(X))
Y_predict=pd.Series(Y_predict)
confusion_matrix=pd.crosstab(Y,Y_predict)
rate1=sum(Y==Y_predict)*1.0/len(Y)# 精确率
print(rate1)
print(confusion_matrix)
'''
weight['组合系数']=(weight['线性回归标准化']+weight['随机森林'])/2

tmp1=pd.DataFrame(np.mean(X,axis=0),index=qlist,columns=[u'满意度'])
weight=weight.join(tmp1)

cmax,cmin=weight['组合系数'].max(),weight['组合系数'].min()
lam=0.5*((tmp1.max()-cmax*tmp1.sum())/(len(tmp1)*cmax-1)+(tmp1.min()-cmin*tmp1.sum())/(len(tmp1)*cmin-1))
tmp2=(tmp1+lam)/(tmp1.sum()+len(tmp1)*lam)
tmp2.rename(columns={u'满意度':u'满意度标准化'},inplace=True)
weight=weight.join(tmp2)

Y_corr=pd.Series([np.corrcoef(X[:,qq],Y)[0,1] for qq in range(len(qlist))],index=qlist)
weight=weight.join(pd.DataFrame(Y_corr,columns=[u'相关系数']))

# 相关系数


'''
for qq in code['Q14']['qlist']:
    print(code['Q14']['code_r'][qq])
    plt.figure(1)
    plt.scatter(d2[qq],d2['Q15'])
    plt.show()
'''

#from scipy.stats import kendalltau
#sns.jointplot(d2[qq],d2['Q15'],kind="hex",stat_func=kendalltau,color="#4CB391")

'''
# 气泡图
for qq in code['Q21']['qlist']:
    print(code['Q21']['code_r'][qq])
    plt.figure(1)
    x=[]
    y=[]
    z=[]
    k=0
    t=pd.crosstab(data[qq],data['Q18']).stack()
    x=[w[0] for w in t.index]
    y=[w[1] for w in t.index]
    z=list(t)
    tt=pd.DataFrame({'x':x,'y':y,'z':z})
    #tt=pd.DataFrame({'x':x,'y':y,'z':z})
    plt.scatter(tt['x'],tt['y'],tt['z']*30,color='orange',alpha=0.6)
    tt.to_csv(qq+'.csv',index=False)
    plt.show()
'''    
