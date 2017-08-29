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
    
'''
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
'''


def code_similar(code1,code2):
    '''
    题目内容相似度用最小编辑距离来度量
    选项相似度分为几种
    1、完全相同：1
    2、单选题：暂时只考虑序号和值都相等的，且共同变量超过一半:2
    2、多选题/排序题：不考虑序号，共同变量超过一半即可：3
    3、矩阵单选题：code_r 暂时只考虑完全匹配
    4、其他情况为0
    
    '''
    code_distance_min=pd.DataFrame(index=code1.keys(),columns=['qnum','similar_content','similar_code'])
    for c1 in code1:
        # 计算题目内容的相似度
        disstance_str=pd.Series(index=code2.keys())
        for c2 in code2:
            if code1[c1]['qtype']==code2[c2]['qtype']:
                disstance_str[c2]=levenshtein(code1[c1]['content'], code2[c2]['content'])
        c2=disstance_str.idxmin()       
        if '%s'%c2 == 'nan':            
            continue
        min_len=(len(code1[c1]['content'])+len(code2[c2]['content']))/2
        similar_content=100-100*disstance_str[c2]/min_len if min_len>0 else 0
        # 计算选项的相似度                                     
        qtype=code2[c2]['qtype']
        if qtype == '单选题':
            t1=code1[c1]['code']
            t2=code2[c2]['code']
            inner_key=list(set(t1.keys())&set(t2.keys()))
            tmp=all([t1[c]==t2[c] for c in inner_key])
            if t1==t2:
                similar_code=1
            elif len(inner_key)>=0.5*len(set(t1.keys())|set(t2.keys())) and tmp:
                similar_code=2
            else:
                similar_code=0
        elif qtype in ['多选题','排序题']:
            t1=code1[c1]['code']
            t2=code2[c2]['code']
            t1=[t1[c] for c in code1[c1]['qlist']]
            t2=[t2[c] for c in code2[c2]['qlist']]
            inner_key=set(t1)&set(t2)
            if t1==t2:
                similar_code=1
            elif len(set(t1)&set(t2))>=0.5*len(set(t1)|set(t2)):
                similar_code=3               
            else:
                similar_code=0
        elif qtype in ['矩阵多选题']:
            t1=code1[c1]['code_r']
            t2=code2[c2]['code_r']
            t1=[t1[c] for c in code1[c1]['qlist']]
            t2=[t2[c] for c in code2[c2]['qlist']]
            inner_key=set(t1)&set(t2)
            if t1==t2:
                similar_code=1
            elif len(set(t1)&set(t2))>=0.5*len(set(t1)|set(t2)):
                similar_code=3               
            else:
                similar_code=0
        elif qtype in ['填空题']:
            similar_code=1
        else:
            similar_code=0

        code_distance_min.loc[c1,'qnum']=c2
        code_distance_min.loc[c1,'similar_content']=similar_content
        code_distance_min.loc[c1,'similar_code']=similar_code


    # 剔除qnum中重复的值
    code_distance_min=code_distance_min.sort_values(['qnum','similar_content','similar_code'],ascending=[False,False,True])
    code_distance_min.loc[code_distance_min.duplicated(['qnum']),:]=np.nan
    code_distance_min=pd.DataFrame(code_distance_min,index=code1.keys())                             
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
        qqlist1=[]
        qqlist2=[]
        code_distance_min=code_similar(code1,code2)
        code1_key=sorted(code1,key=lambda x:int(re.findall('\d+',x)[0]))
        for c1 in code1_key:
            qtype1=code1[c1]['qtype']
            #print('{}:{}'.format(c1,code1[c1]['content']))
            rs_qq=code_distance_min.loc[c1,'qnum']
            similar_content=code_distance_min.loc[c1,'similar_content']
            similar_code=code_distance_min.loc[c1,'similar_code']
            if (similar_content>=similar_threshold) and (similar_code in [1,2]):
                #print('推荐合并第二份数据中的{}({}), 两个题目相似度为为{:.0f}%'.format(rs_qq,code2[rs_qq]['content'],similar))
                print('将自动合并: {} 和 {}'.format(c1,rs_qq))
                user_qq=rs_qq
                qqlist1+=code1[c1]['qlist']
                qqlist2+=code2[user_qq]['qlist']
                qlist1.append(c1)
                qlist2.append(rs_qq)
            elif (similar_content>=similar_threshold) and (similar_code==3):
                # 针对非单选题，此时要调整选项顺序
                t1=code1[c1]['code_r'] if qtype1 =='矩阵单选题' else code1[c1]['code']
                t1_qlist=code1[c1]['qlist']
                t1_value=[t1[k] for k in t1_qlist]
                t2=code2[rs_qq]['code_r'] if qtype1 =='矩阵单选题' else code2[rs_qq]['code']
                t2_qlist=code2[rs_qq]['qlist']
                t2_value=[t2[k] for k in t2_qlist]
                # 保留相同的选项
                t1_qlist_new=[q for q in t1_qlist if t1[q] in list(set(t1_value)&set(t2_value))]
                t2_r=dict(zip([s[1] for s in t2.items()],[s[0] for s in t2.items()]))
                t2_qlist_new=[t2_r[s] for s in [t1[q] for q in t1_qlist_new]]
                code1[c1]['qlist']=t1_qlist_new
                code1[c1]['code']={k:t1[k] for k in t1_qlist_new}
                qqlist1+=t1_qlist_new
                qqlist2+=t2_qlist_new
                qlist1.append(c1)
                qlist2.append(rs_qq)
                print('将自动合并: {} 和 {} (只保留了相同的选项)'.format(c1,rs_qq))
                
            elif similar_code in [1,2]:               
                print('-'*40)
                print('为【  {}:{} 】自动匹配到: '.format(c1,code1[c1]['content']))
                print('  【  {}:{} 】,其相似度为{:.0f}%.'.format(rs_qq,code2[rs_qq]['content'],similar_content))
                tmp=input('是否合并该组题目,请输入 yes/no (也可以输入第二份数据中其他您需要匹配的题目): ')
                tmp=re.sub('\s','',tmp)
                tmp=tmp.lower()
                if tmp in ['yes','y']:
                    user_qq=rs_qq
                elif tmp in ['no','n']:
                    user_qq=None
                else:
                    tmp=re.sub('^q','Q',tmp)
                    if tmp not in code2:
                        user_qq=None
                    elif (tmp in code2) and (tmp!=rs_qq):
                        print('您输入的是{}:{}'.format(tmp,code2[tmp]['content']))
                        user_qq=tmp
                if user_qq==rs_qq:
                    qqlist1+=code1[c1]['qlist']
                    qqlist2+=code2[user_qq]['qlist']
                    qlist1.append(c1)
                    qlist2.append(user_qq)
                    print('将自动合并: {} 和 {}'.format(c1,rs_qq))
                elif user_qq is not None:
                    # 比对两道题目的code
                    if 'code' in code1[c1] and len(code1[c1]['code'])>0:
                        t1=code1[c1]['code_r'] if qtype1 =='矩阵单选题' else code1[c1]['code']
                        t2=code2[user_qq]['code_r'] if code2[user_qq]['qtype'] =='矩阵单选题' else code2[user_qq]['code']
                        if set(t1.values())==set(t2.values()):
                            qqlist1+=code1[c1]['qlist']
                            qqlist2+=code2[user_qq]['qlist']
                            qlist1.append(c1)
                            qlist2.append(user_qq)
                            print('将自动合并: {} 和 {}'.format(c1,user_qq))
                        else:
                            print('两个题目的选项不匹配,将自动跳过.')
                    else:
                        qqlist1+=[code1[c1]['qlist'][0]]
                        qqlist2+=[code2[user_qq]['qlist'][0]]
                        qlist1.append(c1)
                        qlist2.append(user_qq)
                        print('将自动合并: {} 和 {}'.format(c1,user_qq))
                else:
                    print('将自动跳过: {}'.format(c1))
                print('-'*40)
            else:
                print('将自动跳过: {}'.format(c1))
        tmp=input('请问您需要的题目是否都已经合并? 请输入(yes / no)： ')
        tmp=re.sub('\s','',tmp)
        tmp=tmp.lower()
        if tmp in ['no','n']:
            print('请确保接下来您要合并的题目类型和选项完全一样.')
            while 1:
                tmp=input('请输入您想合并的题目对,直接回车则终止输入(如: Q1,Q1 ): ')
                tmp=re.sub('\s','',tmp)# 去掉空格
                tmp=re.sub('，',',',tmp)# 修正可能错误的逗号
                tmp=tmp.split(',')
                tmp=[re.sub('^q','Q',qq) for qq in tmp]
                if len(tmp)<2:
                    break
                if tmp[0] in qlist1 or tmp[1] in qlist2:
                    print('该题已经被合并，请重新输入')
                    continue
                if tmp[0] not in code1 or tmp[1] not in code2:
                    print('输入错误, 请重新输入')
                    continue
                c1=tmp[0]
                c2=tmp[1]
                print('您输入的是：')
                print('第一份数据中的【 {}:{} 】'.format(c1,code1[c1]['content']))
                print('第二份数据中的【 {}:{} 】'.format(c2,code2[c2]['content']))
                w=code_similar({c1:code1[c1]},{c2:code2[c2]})
                similar_code=w.loc[c1,'similar_code']
                if similar_code in [1,2] and len(code1[c1]['qlist'])==len(code2[c2]['qlist']):
                    qqlist1+=code1[c1]['qlist']
                    qqlist2+=code2[c2]['qlist']
                    qlist1.append(c1)
                    qlist2.append(c2)
                    print('将自动合并: {} 和 {}'.format(c1,c2))
                else:
                    print('选项不匹配，请重新输入')                               
        

    else:
        qqlist1=[]
        for qq in qlist1:
            qqlist1=qqlist1+code1[qq]['qlist']
        qqlist2=[]
        for qq in qlist2:
            qqlist2=qqlist2+code2[qq]['qlist']

    # 将题号列表转化成data中的列名
    if mergeqnum in qqlist1:
        mergeqnum=mergeqnum+'merge'
    data1=data1.loc[:,qqlist1]
    data1.loc[:,mergeqnum]=1
    data2=data2.loc[:,qqlist2]
    data2.loc[:,mergeqnum]=2
    
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
'''
ques1=[data1,code1]
ques2=[data2,code2]
qlist1=None
qlist2=None
name1='ques1'
name2='ques2'
mergeqnum='Q0'
similar_threshold=70
'''

data12,code12=data_merge([data1,code1],[data2,code2])









    
    
    
    
