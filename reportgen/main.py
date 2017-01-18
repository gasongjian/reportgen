# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:29:24 2017

@author: 10206913
"""

import pandas as pd
import numpy as np
import report as rpt
import os
import re

filename1=os.path.join('.\\data','All_Data_Readable.xls')
filename2=os.path.join('.\\data','All_Data_Original.xls')
headlen=6

d1=pd.read_excel(filename1,encoding='gbk')
d2=pd.read_excel(filename2,encoding='gbk')
d2.replace({-2:np.nan,-3:np.nan},inplace=True) 

code={}
for name in d1.columns[headlen:]:
    tmp=re.findall(u'^(\d{1,2})[、：:]',name)
    if tmp:
        new_name='Q'+tmp[0]
        current_name='Q'+tmp[0]
        code[new_name]={}
        content=re.findall(u'\d{1,2}[、：:](.*)',name)
        code[new_name]['content']=content[0]
        d1.rename(columns={name:new_name},inplace=True)
        code[new_name]['qlist']=[]
        code[new_name]['code']={}
        code[new_name]['qtype']=''
        code[new_name]['qtype2']=''
        code[new_name]['sample_len']=0
        qcontent=str(d1[new_name])
        if '┋' in qcontent:
            code[new_name]['qtype']='多选题'
        elif '→' in qcontent:
            code[new_name]['qtype']='排序题'
    else:
        tmp2=re.findall(u'^第(\d{1,2})题\(.*?\)',name)
        if tmp2:
            new_name='Q'+tmp2[0]
        else:
            pass
        if new_name not in code.keys():
            j=1
            current_name=new_name
            new_name=new_name+'_R%s'%j
            code[current_name]={}
            code[current_name]['content']=current_name
            code[current_name]['qlist']=[]
            code[current_name]['code']={}
            code[current_name]['code_r']={}
            code[current_name]['qtype']=u'矩阵单选题'
            code[current_name]['qtype2']=''
            code[current_name]['sample_len']=0            
            d1.rename(columns={name:new_name},inplace=True)
        else:
            j+=1
            new_name=new_name+'_R%s'%j
            d1.rename(columns={name:new_name},inplace=True)
        #raise Exception(u"can not dection the NO. of question.")
        #print('can not dection the NO. of question')
        #print(name)
        #pass

for i,name in enumerate(d2.columns[6:]):
    tmp1=re.findall(u'^(\d{1,2})[、：:]',name)
    tmp2=re.findall(u'^第(.*?)题',name)
    if tmp1:    
        current_name='Q'+tmp1[0]# 当前题目的题号
        d2.rename(columns={name:current_name},inplace=True)
        code[current_name]['qlist'].append(current_name)
        code[current_name]['sample_len']=sum(d2[current_name]>=0)
        #code[current_name]['qtype']=u'单选题'
        c1=d1[current_name].unique()
        c2=d2[current_name].unique()
        if c2.dtype != object:
            code[current_name]['code']=dict(zip(c2,c1))
            code[current_name]['qtype']=u'单选题'
        else:
            code[current_name]['qtype']=u'填空题'
    elif tmp2:
        name0='Q'+tmp2[0]
        if name0 != current_name:
            j=1#记录多选题的小题号
            current_name=name0
            c2=list(d2[name].unique())
            print c2
            if code[current_name]['qtype'] == u'矩阵单选题':
                print current_name
                name1='Q'+tmp2[0]+'_R%s'%j
                c1=list(d1[name1].unique())
                code[current_name]['code']=dict(zip(c2,c1))
            else:
                name1='Q'+tmp2[0]+'_A%s'%j            
            code[current_name]['sample_len']=sum(d2[name]>=0)
        else:
            j+=1#记录多选题的小题号
            c2=list(d2[name].unique())
            if code[current_name]['qtype'] == u'矩阵单选题':
                name1='Q'+tmp2[0]+'_R%s'%j
                c1=list(d1[name1].unique())
                old_dict=code[current_name]['code']
                new_dict=dict(zip(c2,c1))
                code[current_name]['code']=old_dict.update(new_dict)
            else:
                name1='Q'+tmp2[0]+'_A%s'%j
        code[current_name]['qlist'].append(name1)
        d2.rename(columns={name:name1},inplace=True)
        tmp3=re.findall(u'第.*?题\((.*?)\)',name)[0]
        if code[current_name]['qtype'] == u'矩阵单选题':
            code[current_name]['code_r'][name1]=tmp3
        else:
            code[current_name]['code'][name1]=tmp3
    else:
        print(d2.columns[i+6])
    # 删除字典中的nan
    if np.nan in  code[current_name]['code']:
        del  code[current_name]['code'][np.nan]
