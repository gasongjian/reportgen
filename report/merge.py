# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:39:21 2017

@author: 10206913
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import report as rpt
from imp import reload
reload(rpt)







def levenshtein(s, t):  
        ''''' From Wikipedia article; Iterative with two matrix rows. '''  
        if s == t: return 0  
        elif len(s) == 0: return len(t)  
        elif len(t) == 0: return len(s)  
        v0 = [None] * (len(t) + 1)  
        v1 = [None] * (len(t) + 1)  
        for i in range(len(v0)):  
            v0[i] = i  
        for i in range(len(s)):  
            v1[0] = i + 1  
            for j in range(len(t)):  
                cost = 0 if s[i] == t[j] else 1  
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)  
            for j in range(len(v0)):  
                v0[j] = v1[j]  
   
        return v1[len(t)]
    
def code_to_text(code): 
    code_key=sorted(code,key=lambda x:int(re.findall('\d+',x)[0]))
    code1={}
    for c  in code_key:
        content=code[c]['content']
        qtype=code[c]['qtype']
        if qtype == '单选题':
            c_key_list=sorted(code[c]['code'])
            c_key_list=list(pd.Series(c_key_list).map(lambda x:int(x) if (('%s'%x!='nan') and not(isinstance(x,str)) and (int(x)==x)) else x))
            value=','.join(['{}:{}'.format(i,code[c]['code'][i]) for i in c_key_list])
        elif qtype in ['多选题','排序题']:
            c_key_list=[code[c]['code'][t] for t in code[c]['qlist'] if t in code[c]['code']]
            value=','.join(['{}:{}'.format(i+1,c_key_list[i]) for i in range(len(c_key_list))])
        elif qtype in ['矩阵多选题']:
            c_key_list=[code[c]['code_r'][t] for t in code[c]['qlist'] if t in code[c]['code_r']]
            value=','.join(['{}:{}'.format(i+1,c_key_list[i]) for i in range(len(c_key_list))])
        else:
            value=''
        value=content+'______'+value
        code1[c]=value
    return  code1             
    

def code_similar(code1,code2):
    code11=code_to_text(code1)
    code22=code_to_text(code2)
    code_distance_min=pd.DataFrame(index=code11.keys(),columns=['qnum','distance','similar'])
    for c1 in code11:
        disstance_str=pd.Series(index=code22.keys())
        for c2 in code22:
            if code1[c1]['qtype']==code2[c2]['qtype']:
                disstance_str[c2]=levenshtein(code11[c1], code22[c2])
        qq=disstance_str.idxmin()
        if '%s'%qq == 'nan':
            distance=np.nan
            rate=np.nan
        else:
            distance=disstance_str[qq]
            rate=100-100*disstance_str[qq]/len(code11[c1]) if len(code11[c1])>0 else 0
        code_distance_min.loc[c1,'qnum']=qq
        code_distance_min.loc[c1,'distance']=distance
        code_distance_min.loc[c1,'similar']=rate
    return code_distance_min                             

def data_merge(ques1,ques2,qlist1=None,qlist2=None,name1='ques1',name2='ques2',\
               mergeqnum='Q0',similar_threshold=70):
    '''合并两份数据
    ques1: 列表,[data1,code1]
    ques2: 列表,[data2,code2]
    '''
    data1,code1=ques1
    data2,code2=ques2

    if (qlist1 is None) or (qlist2 is None):
        qlist1=[]
        qlist2=[]
        code_distance_min=code_similar(code1,code2)
        code1_key=sorted(code1,key=lambda x:int(re.findall('\d+',x)[0]))
        for c1 in code1_key:
            qtype1=code1[c1]['qtype']
            #print('{}:{}'.format(c1,code1[c1]['content']))
            rs_qq=code_distance_min.loc[c1,'qnum']
            similar=code_distance_min.loc[c1,'similar']
            if similar==100:
                #print('推荐合并第二份数据中的{}({}), 两个题目相似度为为{:.0f}%'.format(rs_qq,code2[rs_qq]['content'],similar))
                print('将自动合并: {} 和 {}'.format(c1,rs_qq))
                user_qq=rs_qq
                qlist1.append(c1)
                qlist2.append(user_qq)
            elif similar>=70:
                
                print('-'*40)
                print('正在为第一份数据的  {}:{}  寻找匹配的题目....'.format(c1,code1[c1]['content']))
                print('推荐合并:   {}:{}   ,其相似度为{:.0f}%.'.format(rs_qq,code2[rs_qq]['content'],similar))
                user_qq=input('请输入第二份数据中与 {} 匹配的题号(缺省则表示跳过该题): '.format(c1))
                user_qq=re.sub('\s','',user_qq)
                user_qq=re.sub('^q','Q',user_qq)
                if user_qq not in code2:
                    print('将跳过该题..')
                    user_qq=None
                elif (user_qq in code2) and (user_qq!=rs_qq):
                    print('您输入的是{}:{}'.format(user_qq,code2[user_qq]['content']))
                if user_qq:
                    # 比对两道题目的code
                    if qtype1=='单选题':
                        t1=code1[c1]['code']
                        t2=code2[user_qq]['code']
                        inner_key=list(set(t1.keys())&set(t2.keys()))
                        similar_code=all([t1[c]==t2[c] for c in inner_key])
                    elif qtype1=='填空题':
                        similar_code=True
                    else: 
                        similar_code=False
                    if similar_code:
                        print('将合并第一份数据的{}和第二份数据的{}...'.format(c1,user_qq))
                        qlist1.append(c1)
                        qlist2.append(user_qq)                       
                    else:
                        print('两题选项不匹配，将跳过该题(建议返回重新修改数据)')
                print('-'*40)
            else:
                print('将自动跳过: {}'.format(c1))      

    # 将题号列表转化成data中的列名
    if mergeqnum in qlist1:
        mergeqnum=mergeqnum+'merge'
    qqlist1=[]
    for qq in qlist1:
        qqlist1=qqlist1+code1[qq]['qlist']
    data1=data1[qqlist1]
    data1[mergeqnum]=1
    
    qqlist2=[]
    for qq in qlist2:
        qqlist2=qqlist2+code2[qq]['qlist']
    data2=data2[qqlist2]
    data2[mergeqnum]=2
    
    if len(qqlist1)!=len(qqlist2):
        print('两份数据选项不完全匹配，请检查....')
        raise
    data2=data2.rename(columns=dict(zip(qqlist2,qqlist1)))
    data12=data1.append(data2,ignore_index=True)
    code12={}
    for i,cc in enumerate(qlist1):
        code12[cc]=code1[cc]
        code12[cc]['code'].update(code2[qlist2[i]]['code'])   
    code12[mergeqnum]={'content':u'来源','code':{1:name1,2:name2},'qtype':u'单选题','qlist':[mergeqnum]}
    return data12,code12


data1,code1=rpt.wenjuanxing(['.\\data\\116_113_0.xls','.\\data\\116_113_2.xls'])
data2,code2=rpt.wenjuanxing(['.\\data\\253_250_0.xls','.\\data\\253_250_2.xls'])

data12,code12=data_merge([data1,code1],[data2,code2])








    
    
    
    
