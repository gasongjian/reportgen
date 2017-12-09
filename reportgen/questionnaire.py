# -*- coding: utf-8 -*
'''问卷数据分析工具包
Created on Tue Nov  8 20:05:36 2016
@author: JSong

1、针对问卷星数据，编写并封装了很多常用算法
2、利用report工具包，能将数据直接导出为PPTX

该工具包支持一下功能：
1、编码问卷星、问卷网等数据
2、封装描述统计和交叉分析函数
3、支持生成一份整体的报告和相关数据
'''




import os
import re
import sys
import math
import time



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import report as rpt





#=================================================================
#
#
#                    【问卷数据处理】
#
#
#==================================================================



def read_code(filename):
    '''读取code编码文件并输出为字典格式
    1、支持json格式
    2、支持本包规定的xlsx格式
    see alse to_code
    '''
    file_type=os.path.splitext(filename)[1][1:]
    if file_type == 'json':
        import json
        code=json.load(filename)
        return code
    d=pd.read_excel(filename,header=None)
    d=d[d.any(axis=1)]#去除空行
    d.fillna('NULL',inplace=True)
    d=d.as_matrix()
    code={}
    for i in range(len(d)):
        tmp=d[i,0].strip()
        if tmp == 'key':
            # 识别题号
            code[d[i,1]]={}
            key=d[i,1]
        elif tmp in ['qlist','code_order']:
            # 识别字典值为列表的字段
            ind=np.argwhere(d[i+1:,0]!='NULL')
            if len(ind)>0:
                j=i+1+ind[0][0]
            else:
                j=len(d)
            tmp2=list(d[i:j,1])
            # 列表中字符串的格式化，去除前后空格
            for i in range(len(tmp2)):
                if isinstance(tmp2[i],str):
                    tmp2[i]=tmp2[i].strip()
            code[key][tmp]=tmp2
        elif tmp in ['code','code_r']:
            # 识别字典值为字典的字段
            ind=np.argwhere(d[i+1:,0]!='NULL')
            if len(ind)>0:
                j=i+1+ind[0][0]
            else:
                j=len(d)
            tmp1=list(d[i:j,1])
            tmp2=list(d[i:j,2])
            for i in range(len(tmp2)):
                if isinstance(tmp2[i],str):
                    tmp2[i]=tmp2[i].strip()
            #tmp2=[s.strip() for s in tmp2 if isinstance(s,str) else s]
            code[key][tmp]=dict(zip(tmp1,tmp2))
        # 识别其他的列表字段
        elif (tmp!='NULL') and (d[i,2]=='NULL') and ((i==len(d)-1) or (d[i+1,0]=='NULL')):
            ind=np.argwhere(d[i+1:,0]!='NULL')
            if len(ind)>0:
                j=i+1+ind[0][0]
            else:
                j=len(d)
            if i==len(d)-1:
                code[key][tmp]=d[i,1]
            else:
                tmp2=list(d[i:j,1])
                for i in range(len(tmp2)):
                    if isinstance(tmp2[i],str):
                        tmp2[i]=tmp2[i].strip()
                code[key][tmp]=tmp2
        # 识别其他的字典字段
        elif (tmp!='NULL') and (d[i,2]!='NULL') and ((i==len(d)-1) or (d[i+1,0]=='NULL')):
            ind=np.argwhere(d[i+1:,0]!='NULL')
            if len(ind)>0:
                j=i+1+ind[0][0]
            else:
                j=len(d)
            tmp1=list(d[i:j,1])
            tmp2=list(d[i:j,2])
            for i in range(len(tmp2)):
                if isinstance(tmp2[i],str):
                    tmp2[i]=tmp2[i].strip()
            #tmp2=[s.strip() for s in tmp2 if isinstance(s,str)  else s]
            code[key][tmp]=dict(zip(tmp1,tmp2))
        elif tmp == 'NULL':
            continue
        else:
            code[key][tmp]=d[i,1]
    return code

def save_code(code,filename='code.xlsx'):
    '''code本地输出
    1、输出为json格式，根据文件名自动识别
    2、输出为Excel格式
    see also read_code
    '''
    save_type=os.path.splitext(filename)[1][1:]
    if save_type == 'json':
        code=pd.DataFrame(code)
        code.to_json(filename,force_ascii=False)
        return
    tmp=pd.DataFrame(columns=['name','value1','value2'])
    i=0
    if all(['Q' in c[0] for c in code.keys()]):
        key_qlist=sorted(code,key=lambda c:int(re.findall('\d+',c)[0]))
    else:
        key_qlist=code.keys()
    for key in key_qlist:
        code0=code[key]
        tmp.loc[i]=['key',key,'']
        i+=1
        #print(key)
        for key0 in code0:
            tmp2=code0[key0]
            if (type(tmp2) == list) and tmp2:
                tmp.loc[i]=[key0,tmp2[0],'']
                i+=1
                for ll in tmp2[1:]:
                    tmp.loc[i]=['',ll,'']
                    i+=1
            elif (type(tmp2) == dict) and tmp2:
                try:
                    tmp2_key=sorted(tmp2,key=lambda c:float(re.findall('[\d\.]+','%s'%c)[-1]))
                except:
                    tmp2_key=list(tmp2.keys())
                j=0
                for key1 in tmp2_key:
                    if j==0:
                        tmp.loc[i]=[key0,key1,tmp2[key1]]
                    else:
                        tmp.loc[i]=['',key1,tmp2[key1]]
                    i+=1
                    j+=1
            else:
                if tmp2:
                    tmp.loc[i]=[key0,tmp2,'']
                    i+=1
    if sys.version>'3':
        tmp.to_excel(filename,index=False,header=False)
    else:
        tmp.to_csv(filename,index=False,header=False,encoding='utf-8')


'''问卷数据导入和编码
对每一个题目的情形进行编码：题目默认按照Q1、Q2等给出
Qn.content: 题目内容
Qn.qtype: 题目类型，包含:单选题、多选题、填空题、排序题、矩阵单选题等
Qn.qlist: 题目列表，例如多选题对应着很多小题目
Qn.code: dict,题目选项编码
Qn.code_r: 题目对应的编码(矩阵题目专有)
Qn.code_order: 题目类别的顺序，用于PPT报告的生成[一般后期添加]
Qn.name: 特殊类型，包含：城市题、NPS题等
Qn.weight:dict,每个选项的权重
'''


def dataText_to_code(df,sep,qqlist=None):
    '''编码文本数据

    '''

    if sep in [';','┋']:
        qtype='多选题'
    elif sep in ['-->','→']:
        qtype='排序题'
    if not qqlist:
        qqlist=df.columns
    # 处理多选题
    code={}
    for qq in qqlist:
        tmp=df[qq].map(lambda x : x.split(sep)  if isinstance(x,str) else [])
        item_list=sorted(set(tmp.sum()))
        if qtype == '多选题':
            tmp=tmp.map(lambda x: [int(t in x) for t in item_list])
            code_tmp={'code':{},'qtype':u'多选题','qlist':[],'content':qq}
        elif qtype == '排序题':
            tmp=tmp.map(lambda x:[x.index(t)+1 if t in x else np.nan for t in item_list])
            code_tmp={'code':{},'qtype':u'排序题','qlist':[],'content':qq}
        for i,t in enumerate(item_list):
            column_name='{}_A{:.0f}'.format(qq,i+1)
            df[column_name]=tmp.map(lambda x:x[i])
            code_tmp['code'][column_name]=item_list[i]
            code_tmp['qlist']=code_tmp['qlist']+[column_name]
        code[qq]=code_tmp
        df.drop(qq,axis=1,inplace=True)
    return df,code

def dataCode_to_text(df,code=None):
    '''将按序号数据转换成文本

    '''
    if df.max().max()>1:
        sep='→'
    else:
        sep='┋'
    if code:
        df=df.rename(code)
    qlist=list(df.columns)
    df['text']=np.nan
    if sep in ['┋']:
        for i in df.index:
            w=df.loc[i,:]==1
            df.loc[i,'text']=sep.join(list(w.index[w]))
    elif sep in ['→']:
        for i in df.index:
            w=df.loc[i,:]
            w=w[w>=1].sort_values()
            df.loc[i,'text']=sep.join(list(w.index))
    df.drop(qlist,axis=1,inplace=True)
    return df

def var_combine(data,code,qq1,qq2,sep=',',qnum_new=None,qname_new=None):
    '''将两个变量组合成一个变量
    例如：
    Q1:'性别',Q2: 年龄
    组合后生成：
    1、男_16~19岁
    2、男_20岁~40岁
    3、女_16~19岁
    4、女_20~40岁
    '''
    if qnum_new is None:
        if 'Q'==qq2[0]:
            qnum_new=qq1+'_'+qq2[1:]
        else:
            qnum_new=qq1+'_'+qq2
    if qname_new is None:
        qname_new=code[qq1]['content']+'_'+code[qq2]['content']

    if code[qq1]['qtype']!='单选题' or code[qq2]['qtype']!='单选题':
        print('只支持组合两个单选题，请检查.')
        raise
    d1=data[code[qq1]['qlist'][0]]
    d2=data[code[qq2]['qlist'][0]]
    sm=max(code[qq1]['code'].keys())#  进位制
    sn=max(code[qq2]['code'].keys())# 进位制
    if isinstance(sm,str) or isinstance(sn,str):
        print('所选择的两个变量不符合函数要求.')
        raise
    data[qnum_new]=(d1-1)*sn+d2
    code[qnum_new]={'qtype':'单选题','qlist':[qnum_new],'content':qname_new}

    code_tmp={}
    for c1 in code[qq1]['code']:
        for c2 in code[qq2]['code']:
            cc=(c1-1)*sn+c2
            value='{}{}{}'.format(code[qq1]['code'][c1],sep,code[qq2]['code'][c2])
            code_tmp[cc]=value
    code[qnum_new]['code']=code_tmp
    print('变量已合并，新变量题号为：{}'.format(qnum_new))
    return data,code



def wenjuanwang(filepath='.\\data',encoding='gbk'):
    '''问卷网数据导入和编码
    输入：
    filepath:
        列表，[0]为按文本数据路径，[1]为按序号文本，[2]为编码文件
        文件夹路径，函数会自动在文件夹下搜寻相关数据
    输出：
    (data,code):
        data为按序号的数据，题目都替换成了Q_n
        code为数据编码，可利用函数to_code()导出为json格式或者Excel格式数据
    '''
    if isinstance(filepath,list):
        filename1=filepath[0]
        filename2=filepath[1]
        filename3=filepath[2]
    elif os.path.isdir(filepath):
        filename1=os.path.join(filepath,'All_Data_Readable.csv')
        filename2=os.path.join(filepath,'All_Data_Original.csv')
        filename3=os.path.join(filepath,'code.csv')
    else:
        print('can not dection the filepath!')

    d1=pd.read_csv(filename1,encoding=encoding)
    d1.drop([u'答题时长'],axis=1,inplace=True)
    d2=pd.read_csv(filename2,encoding=encoding)
    d3=pd.read_csv(filename3,encoding=encoding,header=None,na_filter=False)
    d3=d3.as_matrix()
    # 遍历code.csv,获取粗略的编码，暂缺qlist，矩阵单选题的code_r
    code={}
    for i in range(len(d3)):
        if d3[i,0]:
            key=d3[i,0]
            code[key]={}
            code[key]['content']=d3[i,1]
            code[key]['qtype']=d3[i,2]
            code[key]['code']={}
            code[key]['qlist']=[]
        elif d3[i,2]:
            tmp=d3[i,1]
            if code[key]['qtype']  in [u'多选题',u'排序题']:
                tmp=key+'_A'+'%s'%(tmp)
                code[key]['code'][tmp]='%s'%(d3[i,2])
                code[key]['qlist'].append(tmp)
            elif code[key]['qtype']  in [u'单选题']:
                try:
                    tmp=int(tmp)
                except:
                    tmp='%s'%(tmp)
                code[key]['code'][tmp]='%s'%(d3[i,2])
                code[key]['qlist']=[key]
            elif code[key]['qtype']  in [u'填空题']:
                code[key]['qlist']=[key]
            else:
                try:
                    tmp=int(tmp)
                except:
                    tmp='%s'%(tmp)
                code[key]['code'][tmp]='%s'%(d3[i,2])

    # 更新矩阵单选的code_r和qlist
    qnames_Readable=list(d1.columns)
    qnames=list(d2.columns)
    for key in code.keys():
        qlist=[]
        for name in qnames:
            if re.match(key+'_',name) or key==name:
                qlist.append(name)
        if ('qlist' not in code[key]) or (not code[key]['qlist']):
            code[key]['qlist']=qlist
        if code[key]['qtype']  in [u'矩阵单选题']:
            tmp=[qnames_Readable[qnames.index(q)] for q in code[key]['qlist']]
            code_r=[re.findall('_([^_]*?)$',t)[0] for t in tmp]
            code[key]['code_r']=dict(zip(code[key]['qlist'],code_r))
    # 处理时间格式
    d2['start']=pd.to_datetime(d2['start'])
    d2['finish']=pd.to_datetime(d2['finish'])
    tmp=d2['finish']-d2['start']
    tmp=tmp.astype(str).map(lambda x:60*int(re.findall(':(\d+):',x)[0])+int(re.findall(':(\d+)\.',x)[0]))
    ind=np.where(d2.columns=='finish')[0][0]
    d2.insert(int(ind)+1,u'答题时长(秒)',tmp)
    return (d2,code)


def wenjuanxing(filepath='.\\data',headlen=6):
    '''问卷星数据导入和编码
    输入：
    filepath:
        列表, filepath[0]: (23_22_0.xls)为按文本数据路径，filepath[1]: (23_22_2.xls)为按序号文本
        文件夹路径，函数会自动在文件夹下搜寻相关数据，优先为\d+_\d+_0.xls和\d+_\d+_2.xls
    headlen: 问卷星数据基础信息的列数
    输出：
    (data,code):
        data为按序号的数据，题目都替换成了Q_n
        code为数据编码，可利用函数to_code()导出为json格式或者Excel格式数据
    '''
    #filepath='.\\data'
    #headlen=6# 问卷从开始到第一道正式题的数目（一般包含序号，提交答卷时间的等等）
    if isinstance(filepath,list):
        filename1=filepath[0]
        filename2=filepath[1]
    elif os.path.isdir(filepath):
        filelist=os.listdir(filepath)
        n1=n2=0
        for f in filelist:
            s1=re.findall('\d+_\d+_0.xls',f)
            s2=re.findall('\d+_\d+_2.xls',f)
            if s1:
                filename1=s1[0]
                n1+=1
            if s2:
                filename2=s2[0]
                n2+=1
        if n1+n2==0:
            print(u'在文件夹下没有找到问卷星按序号和按文本数据，请检查目录或者工作目录.')
            return
        elif n1+n2>2:
            print(u'存在多组问卷星数据，请检查.')
            return
        filename1=os.path.join(filepath,filename1)
        filename2=os.path.join(filepath,filename2)
    else:
        print('can not dection the filepath!')

    d1=pd.read_excel(filename1)
    d2=pd.read_excel(filename2)
    d2.replace({-2:np.nan,-3:np.nan},inplace=True)
    #d1.replace({u'(跳过)':np.nan},inplace=True)

    code={}
    '''
    遍历一遍按文本数据，获取题号和每个题目的类型
    '''
    for name in d1.columns[headlen:]:
        tmp=re.findall(u'^(\d{1,2})[、：:]',name)
        # 识别多选题、排序题
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
            code[new_name]['name']=''
            qcontent=str(list(d1[new_name]))
            # 单选题和多选题每个选项都可能有开放题，得识别出来
            if ('〖' in qcontent) and ('〗' in qcontent):
                code[new_name]['qlist_open']=[]
            if '┋' in qcontent:
                code[new_name]['qtype']=u'多选题'
            elif '→' in qcontent:
                code[new_name]['qtype']=u'排序题'
        # 识别矩阵单选题
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
                code[current_name]['content']=current_name+'(问卷星数据中未找到题目具体内容)'
                code[current_name]['qlist']=[]
                code[current_name]['code']={}
                code[current_name]['code_r']={}
                code[current_name]['qtype']=u'矩阵单选题'
                code[current_name]['name']=''
                #code[current_name]['sample_len']=0
                d1.rename(columns={name:new_name},inplace=True)
            else:
                j+=1
                new_name=new_name+'_R%s'%j
                d1.rename(columns={name:new_name},inplace=True)
            #raise Exception(u"can not dection the NO. of question.")
            #print('can not dection the NO. of question')
            #print(name)
            #pass
    # 遍历按序号数据，完整编码
    d2qlist=d2.columns[6:].tolist()
    for name in d2qlist:
        tmp1=re.findall(u'^(\d{1,2})[、：:]',name)# 单选题和填空题
        tmp2=re.findall(u'^第(.*?)题',name)# 多选题、排序题和矩阵单选题
        if tmp1:
            current_name='Q'+tmp1[0]# 当前题目的题号
            d2.rename(columns={name:current_name},inplace=True)
            code[current_name]['qlist'].append(current_name)
            #code[current_name]['sample_len']=d2[current_name].count()
            ind=d2[current_name].copy()
            ind=ind.notnull()
            c1=d1.loc[ind,current_name].unique()
            c2=d2.loc[ind,current_name].unique()
            #print('========= %s========'%current_name)
            if (c2.dtype == object) or ((list(c1)==list(c2)) and len(c2)>=min(15,len(d2[ind]))) or (len(c2)>50):
                code[current_name]['qtype']=u'填空题'
            else:
                code[current_name]['qtype']=u'单选题'
                #code[current_name]['code']=dict(zip(c2,c1))
                if 'qlist_open' in code[current_name].keys():
                    tmp=d1[current_name].map(lambda x: re.findall('〖(.*?)〗',x)[0] if re.findall('〖(.*?)〗',x) else '')
                    ind_open=np.argwhere(d2.columns.values==current_name).tolist()[0][0]
                    d2.insert(ind_open+1,current_name+'_open',tmp)
                    d1[current_name]=d1[current_name].map(lambda x: re.sub('〖.*?〗','',x))
                    #c1=d1.loc[ind,current_name].map(lambda x: re.sub('〖.*?〗','',x)).unique()
                    code[current_name]['qlist_open']=[current_name+'_open']
                #c2_tmp=d2.loc[ind,current_name].map(lambda x: int(x) if (('%s'%x!='nan') and not(isinstance(x,str)) and (int(x)==x)) else x)
                code[current_name]['code']=dict(zip(d2.loc[ind,current_name],d1.loc[ind,current_name]))
                #code[current_name]['code']=dict(zip(c2,c1))

        elif tmp2:
            name0='Q'+tmp2[0]
            # 新题第一个选项
            if name0 != current_name:
                j=1#记录多选题的小题号
                current_name=name0
                c2=list(d2[name].unique())

                if code[current_name]['qtype'] == u'矩阵单选题':
                    name1='Q'+tmp2[0]+'_R%s'%j
                    c1=list(d1[name1].unique())
                    code[current_name]['code']=dict(zip(c2,c1))
                    #print(dict(zip(c2,c1)))
                else:
                    name1='Q'+tmp2[0]+'_A%s'%j
                #code[current_name]['sample_len']=d2[name].notnull().sum()
            else:
                j+=1#记录多选题的小题号
                c2=list(d2[name].unique())
                if code[current_name]['qtype'] == u'矩阵单选题':
                    name1='Q'+tmp2[0]+'_R%s'%j
                    c1=list(d1[name1].unique())
                    old_dict=code[current_name]['code'].copy()
                    new_dict=dict(zip(c2,c1))
                    old_dict.update(new_dict)
                    code[current_name]['code']=old_dict.copy()
                else:
                    name1='Q'+tmp2[0]+'_A%s'%j
            code[current_name]['qlist'].append(name1)
            d2.rename(columns={name:name1},inplace=True)
            tmp3=re.findall(u'第.*?题\((.*)\)',name)[0]
            if code[current_name]['qtype'] == u'矩阵单选题':
                code[current_name]['code_r'][name1]=tmp3
            else:
                code[current_name]['code'][name1]=tmp3
            # 识别开放题
            if (code[current_name]['qtype'] == u'多选题'):
                openq=tmp3+'〖.*?〗'
                openq=re.sub('\)','\)',openq)
                openq=re.sub('\(','\(',openq)
                openq=re.compile(openq)
                qcontent=str(list(d1[current_name]))
                if re.findall(openq,qcontent):
                    tmp=d1[current_name].map(lambda x: re.findall(openq,x)[0] if re.findall(openq,x) else '')
                    ind=np.argwhere(d2.columns.values==name1).tolist()[0][0]
                    d2.insert(ind+1,name1+'_open',tmp)
                    code[current_name]['qlist_open'].append(name1+'_open')
        # 删除字典中的nan
        keys=list(code[current_name]['code'].keys())
        for key in keys:
            if '%s'%key == 'nan':
                del  code[current_name]['code'][key]

    # 处理一些特殊题目，给它们的选项固定顺序，例如年龄、收入等
    for k in code.keys():
        content=code[k]['content']
        qtype=code[k]['qtype']
        if ('code' in code[k]) and (code[k]['code']!={}):
            tmp1=code[k]['code'].keys()
            tmp2=code[k]['code'].values()
            # 识别选项是否是有序变量
            tmp3=[len(re.findall('\d+','%s'%v))>0 for v in tmp2]#是否有数字
            tmp4=[len(re.findall('-|~','%s'%v))>0 for v in tmp2]#是否有"-"或者"~"
            if (np.array(tmp3).sum()>=len(tmp2)-2) or (np.array(tmp4).sum()>=len(tmp2)*0.8-(1e-17)):
                try:
                    tmp_key=sorted(code[k]['code'],key=lambda c:float(re.findall('[\d\.]+','%s'%c)[-1]))
                except:
                    tmp_key=list(tmp1)
                code_order=[code[k]['code'][v] for v in tmp_key]
                code[k]['code_order']=code_order
            # 识别矩阵量表题
            if qtype=='矩阵单选题':
                tmp3=[int(re.findall('\d+','%s'%v)[0]) for v in tmp2 if re.findall('\d+','%s'%v)]
                if (set(tmp3)<=set([0,1,2,3,4,5,6,7,8,9,10])) and (len(tmp3)==len(tmp2)):
                    code[k]['weight']=dict(zip(tmp1,tmp3))
                    continue
            # 识别特殊题型
            if ('性别' in content) and ('男' in tmp2) and ('女' in tmp2):
                code[k]['name']='性别'
            if ('gender' in content.lower()) and ('Male' in tmp2) and ('Female' in tmp2):
                code[k]['name']='性别'
            if (('年龄' in content) or ('age' in content.lower())) and (np.array(tmp3).sum()>=len(tmp2)-1):
                code[k]['name']='年龄'
            if ('满意度' in content) and ('整体' in content):
                tmp3=[int(re.findall('\d+','%s'%v)[0]) for v in tmp2 if re.findall('\d+','%s'%v)]
                if set(tmp3)<=set([0,1,2,3,4,5,6,7,8,9,10]):
                    code[k]['name']='满意度'
                    if len(tmp3)==len(tmp2):
                        code[k]['weight']=dict(zip(tmp1,tmp3))
            if ('意愿' in content) and ('推荐' in content):
                tmp3=[int(re.findall('\d+','%s'%v)[0]) for v in tmp2 if re.findall('\d+','%s'%v)]
                if set(tmp3)<=set([0,1,2,3,4,5,6,7,8,9,10]):
                    code[k]['name']='NPS'
                    if len(tmp3)==len(tmp2):
                        weight=pd.Series(dict(zip(tmp1,tmp3)))
                        weight=weight.replace(dict(zip([0,1,2,3,4,5,6,7,8,9,10],[-100,-100,-100,-100,-100,-100,-100,0,0,100,100])))
                        code[k]['weight']=weight.to_dict()

    try:
        d2[u'所用时间']=d2[u'所用时间'].map(lambda s: int(s[:-1]))
    except:
        pass

    return (d2,code)


def load_data(method='filedialog',**kwargs):
    '''导入问卷数据
    # 暂时只支持已编码的和问卷星数据
    1、支持路径搜寻
    2、支持自由选择文件
    method:
        -filedialog: 打开文件窗口选择
        -pathsearch：自带搜索路径，需提供filepath
    '''
    if method=='filedialog':
        import tkinter as tk
        from tkinter.filedialog import askopenfilenames
        tk.Tk().withdraw();
        #print(u'请选择编码所需要的数据文件（支持问卷星和已编码好的数据）')
        if 'initialdir' in kwargs:
            initialdir=kwargs['initialdir']
        elif os.path.isdir('.\\data'):
            initialdir = ".\\data"
        else:
            initialdir = "."
        title =u"请选择编码所需要的数据文件（支持问卷星和已编码好的数据）"
        filetypes = (("Excel files","*.xls;*.xlsx"),("CSV files","*.csv"),("all files","*.*"))
        filenames=[]
        while len(filenames)<1:
            filenames=askopenfilenames(initialdir=initialdir,title=title,filetypes=filetypes)
            if len(filenames)<1:
                print('请至少选择一个文件.')
        filenames=list(filenames)
    elif method == 'pathsearch':
        if 'filepath' in kwargs:
            filepath=kwargs['filepath']
        else :
            filepath='.\\data\\'
        if os.path.isdir(filepath):
            filenames=os.listdir(filepath)
            filenames=[os.path.join(filepath,s) for s in filenames]
        else:
            print('搜索路径错误')
            raise
    info=[]
    for filename in filenames:
        filename_nopath=os.path.split(filename)[1]
        data=read_data(filename)
        # 第一列包含的字段
        field_c1=set(data.iloc[:,0].dropna().unique())
        field_r1=set(data.columns)
        # 列名是否包含Q
        hqlen=[len(re.findall('^[qQ]\d+',c))>0 for c in field_r1]
        hqrate=hqlen.count(True)/len(field_r1) if len(field_r1)>0 else 0
        rowlens,collens=data.shape
        # 数据中整数/浮点数的占比
        rate_real=data.applymap(lambda x:isinstance(x,(int,float))).sum().sum()/rowlens/collens

        tmp={'filename':filename_nopath,'filenametype':'','rowlens':rowlens,'collens':collens,\
        'field_c1':field_c1,'field_r1':field_r1,'type':'','rate_real':rate_real}

        if len(re.findall('^data.*\.xls',filename_nopath))>0:
            tmp['filenametype']='data'
        elif len(re.findall('^code.*\.xls',filename_nopath))>0:
            tmp['filenametype']='code'
        elif len(re.findall('\d+_\d+_\d.xls',filename_nopath))>0:
            tmp['filenametype']='wenjuanxing'

        if tmp['filenametype']=='code' or set(['key','code','qlist','qtype']) < field_c1:
            tmp['type']='code'
        if tmp['filenametype']=='wenjuanxing' or len(set(['序号','提交答卷时间','所用时间','来自IP','来源','来源详情','总分'])&field_r1)>=5:
            tmp['type']='wenjuanxing'
        if tmp['filenametype']=='data' or hqrate>=0.5:
            tmp['type']='data'
        info.append(tmp)
    questype=[k['type'] for k in info]
    # 这里有一个优先级存在，优先使用已编码好的数据，其次是问卷星数据
    if questype.count('data')*questype.count('code')==1:
        data=read_data(filenames[questype.index('data')])
        code=read_code(filenames[questype.index('code')])
    elif questype.count('wenjuanxing')>=2:
        filenames=[(f,info[i]['rate_real']) for i,f in enumerate(filenames) if questype[i]=='wenjuanxing']
        tmp=[]
        for f,rate_real in filenames:
            t2=0 if rate_real<0.5 else 2
            d=pd.read_excel(f)
            d=d.iloc[:,0]
            tmp.append((t2,d))
            #print('添加{}'.format(t2))
            tmp_equal=0
            for t,d0 in tmp[:-1]:
                if len(d)==len(d0)  and all(d==d0):
                    tmp_equal+=1
                    tmp[-1]=(t2+int(t/10)*10,tmp[-1][1])
            max_quesnum=max([int(t/10) for t,d in tmp])
            if tmp_equal==0:
                tmp[-1]=(tmp[-1][0]+max_quesnum*10+10,tmp[-1][1])
            #print('修改为{}'.format(tmp[-1][0]))
        # 重新整理所有的问卷数据
        questype=[t for t,d in tmp]
        filenames=[f for f,r in filenames]
        quesnums=max([int(t/10) for t in questype])#可能存在的数据组数
        filename_wjx=[]
        for i in range(1,quesnums+1):
            if questype.count(i*10)==1 and questype.count(i*10+2)==1:
                filename_wjx.append([filenames[questype.index(i*10)],filenames[questype.index(i*10+2)]])
        if len(filename_wjx)==1:
            data,code=wenjuanxing(filename_wjx[0])
        elif len(filename_wjx)>1:
            print('脚本识别出多组问卷星数据，请选择需要编码的数据：')
            for i,f in enumerate(filename_wjx):
                print('{}: {}'.format(i+1,'/'.join([os.path.split(f[0])[1],os.path.split(f[1])[1]])))
            ii=input('您选择的数据是(数据前的编码，如：1):')
            ii=re.sub('\s','',ii)
            if ii.isnumeric():
                data,code=wenjuanxing(filename_wjx[int(ii)-1])
            else:
                print('您输入正确的编码.')
        else:
            print('没有找到任何问卷数据..')
            raise
    else:
        print('没有找到任何数据')
        raise
    return data,code




def spec_rcode(data,code):
    city={'北京':0,'上海':0,'广州':0,'深圳':0,'成都':1,'杭州':1,'武汉':1,'天津':1,'南京':1,'重庆':1,'西安':1,'长沙':1,'青岛':1,'沈阳':1,'大连':1,'厦门':1,'苏州':1,'宁波':1,'无锡':1,\
    '福州':2,'合肥':2,'郑州':2,'哈尔滨':2,'佛山':2,'济南':2,'东莞':2,'昆明':2,'太原':2,'南昌':2,'南宁':2,'温州':2,'石家庄':2,'长春':2,'泉州':2,'贵阳':2,'常州':2,'珠海':2,'金华':2,\
    '烟台':2,'海口':2,'惠州':2,'乌鲁木齐':2,'徐州':2,'嘉兴':2,'潍坊':2,'洛阳':2,'南通':2,'扬州':2,'汕头':2,'兰州':3,'桂林':3,'三亚':3,'呼和浩特':3,'绍兴':3,'泰州':3,'银川':3,'中山':3,\
    '保定':3,'西宁':3,'芜湖':3,'赣州':3,'绵阳':3,'漳州':3,'莆田':3,'威海':3,'邯郸':3,'临沂':3,'唐山':3,'台州':3,'宜昌':3,'湖州':3,'包头':3,'济宁':3,'盐城':3,'鞍山':3,'廊坊':3,'衡阳':3,\
    '秦皇岛':3,'吉林':3,'大庆':3,'淮安':3,'丽江':3,'揭阳':3,'荆州':3,'连云港':3,'张家口':3,'遵义':3,'上饶':3,'龙岩':3,'衢州':3,'赤峰':3,'湛江':3,'运城':3,'鄂尔多斯':3,'岳阳':3,'安阳':3,\
    '株洲':3,'镇江':3,'淄博':3,'郴州':3,'南平':3,'齐齐哈尔':3,'常德':3,'柳州':3,'咸阳':3,'南充':3,'泸州':3,'蚌埠':3,'邢台':3,'舟山':3,'宝鸡':3,'德阳':3,'抚顺':3,'宜宾':3,'宜春':3,'怀化':3,\
    '榆林':3,'梅州':3,'呼伦贝尔':3,'临汾':4,'南阳':4,'新乡':4,'肇庆':4,'丹东':4,'德州':4,'菏泽':4,'九江':4,'江门市':4,'黄山':4,'渭南':4,'营口':4,'娄底':4,'永州市':4,'邵阳':4,'清远':4,\
    '大同':4,'枣庄':4,'北海':4,'丽水':4,'孝感':4,'沧州':4,'马鞍山':4,'聊城':4,'三明':4,'开封':4,'锦州':4,'汉中':4,'商丘':4,'泰安':4,'通辽':4,'牡丹江':4,'曲靖':4,'东营':4,'韶关':4,'拉萨':4,\
    '襄阳':4,'湘潭':4,'盘锦':4,'驻马店':4,'酒泉':4,'安庆':4,'宁德':4,'四平':4,'晋中':4,'滁州':4,'衡水':4,'佳木斯':4,'茂名':4,'十堰':4,'宿迁':4,'潮州':4,'承德':4,'葫芦岛':4,'黄冈':4,'本溪':4,\
    '绥化':4,'萍乡':4,'许昌':4,'日照':4,'铁岭':4,'大理州':4,'淮南':4,'延边州':4,'咸宁':4,'信阳':4,'吕梁':4,'辽阳':4,'朝阳':4,'恩施州':4,'达州市':4,'益阳市':4,'平顶山':4,'六安':4,'延安':4,\
    '梧州':4,'白山':4,'阜阳':4,'铜陵市':4,'河源':4,'玉溪市':4,'黄石':4,'通化':4,'百色':4,'乐山市':4,'抚州市':4,'钦州':4,'阳江':4,'池州市':4,'广元':4,'滨州':5,'阳泉':5,'周口市':5,'遂宁':5,\
    '吉安':5,'长治':5,'铜仁':5,'鹤岗':5,'攀枝花':5,'昭通':5,'云浮':5,'伊犁州':5,'焦作':5,'凉山州':5,'黔西南州':5,'广安':5,'新余':5,'锡林郭勒':5,'宣城':5,'兴安盟':5,'红河州':5,'眉山':5,\
    '巴彦淖尔':5,'双鸭山市':5,'景德镇市':5,'鸡西':5,'三门峡':5,'宿州':5,'汕尾':5,'阜新':5,'张掖':5,'玉林':5,'乌兰察布':5,'鹰潭':5,'黑河':5,'伊春':5,'贵港市':5,'漯河':5,'晋城':5,'克拉玛依':5,\
    '随州':5,'保山':5,'濮阳':5,'文山州':5,'嘉峪关':5,'六盘水':5,'乌海':5,'自贡':5,'松原':5,'内江':5,'黔东南州':5,'鹤壁':5,'德宏州':5,'安顺':5,'资阳':5,'鄂州':5,'忻州':5,'荆门':5,'淮北':5,\
    '毕节':5,'巴音郭楞':5,'防城港':5,'天水':5,'黔南州':5,'阿坝州':5,'石嘴山':5,'安康':5,'亳州市':5,'昌吉州':5,'普洱':5,'楚雄州':5,'白城':5,'贺州':5,'哈密':5,'来宾':5,'庆阳':5,'河池':5,\
    '张家界 雅安':5,'辽源':5,'湘西州':5,'朔州':5,'临沧':5,'白银':5,'塔城地区':5,'莱芜':5,'迪庆州':5,'喀什地区':5,'甘孜州':5,'阿克苏':5,'武威':5,'巴中':5,'平凉':5,'商洛':5,'七台河':5,'金昌':5,\
    '中卫':5,'阿勒泰':5,'铜川':5,'海西州':5,'吴忠':5,'固原':5,'吐鲁番':5,'阿拉善盟':5,'博尔塔拉州':5,'定西':5,'西双版纳':5,'陇南':5,'大兴安岭':5,'崇左':5,'日喀则':5,'临夏州':5,'林芝':5,\
    '海东':5,'怒江州':5,'和田地区':5,'昌都':5,'儋州':5,'甘南州':5,'山南':5,'海南州':5,'海北州':5,'玉树州':5,'阿里地区':5,'那曲地区':5,'黄南州':5,'克孜勒苏州':5,'果洛州':5,'三沙':5}
    code_keys=list(code.keys())
    for qq in code_keys:
        qlist=code[qq]['qlist']
        #qtype=code[qq]['qtype']
        content=code[qq]['content']
        ind=list(data.columns).index(qlist[-1])
        data1=data[qlist]
        '''
        识别问卷星中的城市题
        '''
        tf1=u'城市' in content
        tf2=data1[data1.notnull()].applymap(lambda x:'-' in '%s'%x).all().all()
        tf3=(qq+'a' not in data.columns) and (qq+'b' not in data.columns)
        if tf1 and tf2 and tf3:
            # 省份和城市
            tmp1=data[qq].map(lambda x:x.split('-')[0])
            tmp2=data[qq].map(lambda x:x.split('-')[1])
            tmp2[tmp1==u'上海']=u'上海'
            tmp2[tmp1==u'北京']=u'北京'
            tmp2[tmp1==u'天津']=u'天津'
            tmp2[tmp1==u'重庆']=u'重庆'
            tmp2[tmp1==u'香港']=u'香港'
            tmp2[tmp1==u'澳门']=u'澳门'
            data.insert(ind+1,qq+'a',tmp1)
            data.insert(ind+2,qq+'b',tmp2)
            code[qq+'a']={'content':'省份','qtype':'填空题','qlist':[qq+'a']}
            code[qq+'b']={'content':'城市','qtype':'填空题','qlist':[qq+'b']}
            tmp3=data[qq+'b'].map(lambda x: city[x] if x in city.keys() else x)
            tmp3=tmp3.map(lambda x: 6 if isinstance(x,str) else x)
            data.insert(ind+3,qq+'c',tmp3)
            code[qq+'c']={'content':'城市分级','qtype':'单选题','qlist':[qq+'c'],\
            'code':{0:'北上广深',1:'新一线',2:'二线',3:'三线',4:'四线',5:'五线',6:'五线以下'}}

    return data,code


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
        if 'code' in code1[cc] and 'code' in code2[qlist2[i]]:
            code12[cc]['code'].update(code2[qlist2[i]]['code'])
    code12[mergeqnum]={'content':u'来源','code':{1:name1,2:name2},'qtype':u'单选题','qlist':[mergeqnum]}
    return data12,code12









## ===========================================================
#
#
#                     数据清洗                          #
#
#
## ==========================================================




def clean_ftime(ftime,cut_percent=0.25):
    '''
    ftime 是完成问卷的秒数
    思路：
    1、只考虑截断问卷完成时间较小的样本
    2、找到完成时间变化的拐点，即需要截断的时间点
    返回：r
    建议截断<r的样本
    '''
    t_min=int(ftime.min())
    t_cut=int(ftime.quantile(cut_percent))
    x=np.array(range(t_min,t_cut))
    y=np.array([len(ftime[ftime<=i]) for i in range(t_min,t_cut)])
    z1 = np.polyfit(x, y, 4) # 拟合得到的函数
    z2=np.polyder(z1,2) #求二阶导数
    r=np.roots(np.polyder(z2,1))
    r=int(r[0])
    return r



## ===========================================================
#
#
#                     数据分析和输出                          #
#
#
## ==========================================================


def data_auto_code(data):
    '''智能判断问卷数据
    输入
    data: 数据框，列名需要满足Qi或者Qi_
    输出：
    code: 自动编码
    '''
    data=pd.DataFrame(data)
    columns=data.columns
    columns=[c for c in columns if re.match('Q\d+',c)]
    code={}
    for cc in columns:
        # 识别题目号
        if '_' not in cc:
            key=cc
        else:
            key=cc.split('_')[0]
        # 新的题目则产生新的code
        if key not in code:
            code[key]={}
            code[key]['qlist']=[]
            code[key]['code']={}
            code[key]['content']=key
            code[key]['qtype']=''
        # 处理各题目列表
        if key == cc:
            code[key]['qlist']=[key]
        elif re.findall('^'+key+'_[a-zA-Z]{0,}\d+$',cc):
            code[key]['qlist'].append(cc)
        else:
            if 'qlist_open' in code[key]:
                code[key]['qlist_open'].append(cc)
            else:
                code[key]['qlist_open']=[cc]

    for kk in code.keys():
        dd=data[code[kk]['qlist']]
        # 单选题和填空题
        if len(dd.columns)==1:
            tmp=dd[dd.notnull()].iloc[:,0].unique()
            if dd.iloc[:,0].value_counts().mean() >=2:
                code[kk]['qtype']=u'单选题'
                code[kk]['code']=dict(zip(tmp,tmp))
            else:
                code[kk]['qtype']=u'填空题'
                del code[kk]['code']
        else:
            tmp=set(dd[dd.notnull()].as_matrix().flatten())
            if set(tmp)==set([0,1]):
                code[kk]['qtype']=u'多选题'
                code[kk]['code']=dict(zip(code[kk]['qlist'],code[kk]['qlist']))
            elif 'R' in code[kk]['qlist'][0]:
                code[kk]['qtype']=u'矩阵单选题'
                code[kk]['code_r']=dict(zip(code[kk]['qlist'],code[kk]['qlist']))
                code[kk]['code']=dict(zip(list(tmp),list(tmp)))
            else:
                code[kk]['qtype']=u'排序题'
                code[kk]['code']=dict(zip(code[kk]['qlist'],code[kk]['qlist']))
    return code




def save_data(data,filename=u'data.xlsx',code=None):
    '''保存问卷数据到本地
    根据filename后缀选择相应的格式保存
    如果有code,则保存按文本数据
    '''
    savetype=os.path.splitext(filename)[1][1:]
    data1=data.copy()
    if code:
        for qq in code.keys():
            qtype=code[qq]['qtype']
            qlist=code[qq]['qlist']
            if qtype == u'单选题':
                # 将序号换成文本，题号加上具体内容
                data1[qlist[0]].replace(code[qq]['code'],inplace=True)
                data1.rename(columns={qq:'{}({})'.format(qq,code[qq]['content'])},inplace=True)
            elif qtype == u'矩阵单选题':
                # 同单选题
                data1[code[qq]['qlist']].replace(code[qq]['code'],inplace=True)
                tmp1=code[qq]['qlist']
                tmp2=['{}({})'.format(q,code[qq]['code_r'][q]) for q in tmp1]
                data1.rename(columns=dict(zip(tmp1,tmp2)),inplace=True)
            elif qtype in [u'排序题']:
                # 先变成一道题，插入表中，然后再把序号变成文本
                tmp=data[qlist]
                tmp=tmp.rename(columns=code[qq]['code'])
                tmp=dataCode_to_text(tmp)
                ind=list(data1.columns).index(qlist[0])
                qqname='{}({})'.format(qq,code[qq]['content'])
                data1.insert(ind,qqname,tmp)

                tmp1=code[qq]['qlist']
                tmp2=['{}_{}'.format(qq,code[qq]['code'][q]) for q in tmp1]
                data1.rename(columns=dict(zip(tmp1,tmp2)),inplace=True)
            elif qtype in [u'多选题']:
                # 先变成一道题，插入表中，然后再把序号变成文本
                tmp=data[qlist]
                tmp=tmp.rename(columns=code[qq]['code'])
                tmp=dataCode_to_text(tmp)
                ind=list(data1.columns).index(qlist[0])
                qqname='{}({})'.format(qq,code[qq]['content'])
                data1.insert(ind,qqname,tmp)

                for q in qlist:
                    data1[q].replace({0:'',1:code[qq]['code'][q]},inplace=True)
                tmp2=['{}_{}'.format(qq,code[qq]['code'][q]) for q in qlist]
                data1.rename(columns=dict(zip(qlist,tmp2)),inplace=True)

            else:
                data1.rename(columns={qq:'{}({})'.format(qq,code[qq]['content'])},inplace=True)
    if (savetype == u'xlsx') or (savetype == u'xls'):
        data1.to_excel(filename,index=False)
    elif savetype == u'csv':
        data1.to_csv(filename,index=False)


def read_data(filename):
    savetype=os.path.splitext(filename)[1][1:]
    if (savetype==u'xlsx') or (savetype==u'xls'):
        data=pd.read_excel(filename)
    elif savetype==u'csv':
        data=pd.read_csv(filename)
    else:
        print('con not read file!')
    return data



def sa_to_ma(data):
    '''单选题数据转换成多选题数据
    data是单选题数据, 要求非有效列别为nan
    可以使用内置函数pd.get_dummies()代替
    '''
    if isinstance(data,pd.core.frame.DataFrame):
        data=data[data.columns[0]]
    #categorys=sorted(data[data.notnull()].unique())
    categorys=data[data.notnull()].unique()
    try:
        categorys=sorted(categorys)
    except:
        pass
        #print('sa_to_ma function::cannot sorted')
    data_ma=pd.DataFrame(index=data.index,columns=categorys)
    for c in categorys:
        data_ma[c]=data.map(lambda x : int(x==c))
    data_ma.loc[data.isnull(),:]=np.nan
    return data_ma

def to_dummpy(data,code,qqlist=None,qtype_new='多选题',ignore_open=True):
    '''转化成哑变量
    将数据中所有的单选题全部转化成哑变量，另外剔除掉开放题和填空题
    返回一个很大的只有0和1的数据
    '''
    if qqlist is None:
        qqlist=sorted(code,key=lambda x:int(re.findall('\d+',x)[0]))
    bdata=pd.DataFrame()
    bcode={}
    for qq in qqlist:
        qtype=code[qq]['qtype']
        data0=data[code[qq]['qlist']]
        if qtype=='单选题':
            data0=data0.iloc[:,0]
            categorys=data0[data0.notnull()].unique()
            try:
                categorys=sorted(categorys)
            except :
                pass
            categorys=[t for t in categorys if t in code[qq]['code']]
            cname=[code[qq]['code'][k] for k in categorys]
            columns_name=['{}_A{}'.format(qq,i+1) for i in range(len(categorys))]
            tmp=pd.DataFrame(index=data0.index,columns=columns_name)
            for i,c in enumerate(categorys):
                tmp[columns_name[i]]=data0.map(lambda x : int(x==c))
            #tmp.loc[data0.isnull(),:]=0
            code_tmp={'content':code[qq]['content'],'qtype':qtype_new}
            code_tmp['code']=dict(zip(columns_name,cname))
            code_tmp['qlist']=columns_name
            bcode.update({qq:code_tmp})
            bdata=pd.concat([bdata,tmp],axis=1)
        elif qtype in ['多选题','排序题','矩阵单选题']:
            bdata=pd.concat([bdata,data0],axis=1)
            bcode.update({qq:code[qq]})
    bdata=bdata.fillna(0)
    try:
        bdata=bdata.astype(np.int64,raise_on_error=False)
    except :
        pass
    return bdata,bcode


def qdata_flatten(data,code,quesid=None,userid_begin=None):
    '''将问卷数据展平，字段如下
    userid: 用户ID
    quesid: 问卷ID
    qnum: 题号
    qname: 题目内容
    qtype: 题目类型
    samplelen:题目的样本数
    itemnum: 选项序号
    itemname: 选项内容
    code: 用户的选择
    codename: 用户选择的具体值
    count: 计数
    percent(%): 计数占比（百分比）
    '''

    if not userid_begin:
        userid_begin=1000000
    data.index=[userid_begin+i+1 for i in range(len(data))]
    if '提交答卷时间' in data.columns:
        begin_date=pd.to_datetime(data['提交答卷时间']).min().strftime('%Y-%m-%d')
        end_date=pd.to_datetime(data['提交答卷时间']).max().strftime('%Y-%m-%d')
    else:
        begin_date=''
        end_date=''
    data,code=to_dummpy(data,code,qtype_new='单选题')
    code_item={}
    for qq in code:
        if code[qq]['qtype']=='矩阵单选题':
            code_item.update(code[qq]['code_r'])
        else :
            code_item.update(code[qq]['code'])

    qdata=data.stack().reset_index()
    qdata.columns=['userid','qn_an','code']
    qdata['qnum']=qdata['qn_an'].map(lambda x:x.split('_')[0])
    qdata['itemnum']=qdata['qn_an'].map(lambda x:'_'.join(x.split('_')[1:]))

    if quesid:
        qdata['quesid']=quesid
        qdata=qdata[['userid','quesid','qnum','itemnum','code']]
    else:
        qdata=qdata[['userid','qnum','itemnum','code']]
    #  获取描述统计信息:
    samplelen=qdata.groupby(['userid','qnum'])['code'].sum().map(lambda x:int(x>0)).unstack().sum()
    quesinfo=qdata.groupby(['qnum','itemnum','code'])['code'].count()
    quesinfo.name='count'
    quesinfo=quesinfo.reset_index()
    quesinfo=quesinfo[quesinfo['code']!=0]
    #quesinfo=qdata.groupby(['quesid','qnum','itemnum'])['code'].sum()
    quesinfo['samplelen']=quesinfo['qnum'].replace(samplelen.to_dict())
    quesinfo['percent(%)']=0
    quesinfo.loc[quesinfo['samplelen']>0,'percent(%)']=100*quesinfo.loc[quesinfo['samplelen']>0,'count']/quesinfo.loc[quesinfo['samplelen']>0,'samplelen']

    quesinfo['qname']=quesinfo['qnum'].map(lambda x: code[x]['content'])
    quesinfo['qtype']=quesinfo['qnum'].map(lambda x: code[x]['qtype'])
    quesinfo['itemname']=quesinfo['qnum']+quesinfo['itemnum'].map(lambda x:'_%s'%x)
    quesinfo['itemname']=quesinfo['itemname'].replace(code_item)
    #quesinfo['itemname']=quesinfo['qn_an'].map(lambda x: code[x.split('_')[0]]['code_r'][x] if \
     #code[x.split('_')[0]]['qtype']=='矩阵单选题' else code[x.split('_')[0]]['code'][x])
    # 各个选项的含义
    quesinfo['codename']=''
    quesinfo.loc[quesinfo['code']==0,'codename']='否'
    quesinfo.loc[quesinfo['code']==1,'codename']='是'
    quesinfo['tmp']=quesinfo['qnum']+quesinfo['code'].map(lambda x:'_%s'%int(x))
    quesinfo['codename'].update(quesinfo.loc[(quesinfo['code']>0)&(quesinfo['qtype']=='矩阵单选题'),'tmp']\
    .map(lambda x: code[x.split('_')[0]]['code'][int(x.split('_')[1])]))
    quesinfo['codename'].update(quesinfo.loc[(quesinfo['code']>0)&(quesinfo['qtype']=='排序题'),'tmp'].map(lambda x: 'Top{}'.format(x.split('_')[1])))
    quesinfo['begin_date']=begin_date
    quesinfo['end_date']=end_date
    if quesid:
        quesinfo['quesid']=quesid
        quesinfo=quesinfo[['quesid','begin_date','end_date','qnum','qname','qtype','samplelen','itemnum','itemname','code','codename','count','percent(%)']]
    else:
        quesinfo=quesinfo[['qnum','qname','qtype','samplelen','itemnum','itemname','code','codename','count','percent(%)']]

    # 排序
    quesinfo['qnum']=quesinfo['qnum'].astype('category')
    quesinfo['qnum'].cat.set_categories(sorted(list(quesinfo['qnum'].unique()),key=lambda x:int(re.findall('\d+',x)[0])), inplace=True)
    quesinfo['itemnum']=quesinfo['itemnum'].astype('category')
    quesinfo['itemnum'].cat.set_categories(sorted(list(quesinfo['itemnum'].unique()),key=lambda x:int(re.findall('\d+',x)[0])), inplace=True)
    quesinfo=quesinfo.sort_values(['qnum','itemnum','code'])
    return qdata,quesinfo




def confidence_interval(p,n,alpha=0.05):
    import scipy.stats as stats
    t=stats.norm.ppf(1-alpha/2)
    ci=t*math.sqrt(p*(1-p)/n)
    #a=p-stats.norm.ppf(1-alpha/2)*math.sqrt(p*(1-p)/n)
    #b=p+stats.norm.ppf(1-alpha/2)*math.sqrt(p*(1-p)/n)
    return ci

def sample_size_cal(interval,N,alpha=0.05):
    '''调研样本量的计算
    参考：https://www.surveysystem.com/sscalc.htm
    sample_size_cal(interval,N,alpha=0.05)
    输入：
    interval: 误差范围,例如0.03
    N: 总体的大小,一般1万以上就没啥差别啦
    alpha：置信水平，默认95%
    '''
    import scipy.stats as stats
    p=stats.norm.ppf(1-alpha/2)
    if interval>1:
        interval=interval/100
    samplesize=p**2/4/interval**2
    if N:
        samplesize=samplesize*N/(samplesize+N)
    samplesize=int(round(samplesize))
    return samplesize


def gof_test(fo,fe=None,alpha=0.05):
    '''拟合优度检验
    输入：
    fo:观察频数
    fe:期望频数，缺省为平均数
    返回：
    1: 样本与总体有差异
    0：样本与总体无差异
    例子：
    gof_test(np.array([0.3,0.4,0.3])*222)
    '''
    import scipy.stats as stats
    fo=np.array(fo).flatten()
    C=len(fo)
    if not fe:
        N=fo.sum()
        fe=np.array([N/C]*C)
    else:
        fe=np.array(fe).flatten()
    chi_value=(fo-fe)**2/fe
    chi_value=chi_value.sum()
    chi_value_fit=stats.chi2.ppf(q=1-alpha,df=C-1)
    #CV=np.sqrt((fo-fe)**2/fe**2/(C-1))*100
    if chi_value>chi_value_fit:
        result=1
    else:
        result=0
    return result


def chi2_test(fo,alpha=0.05):
    import scipy.stats as stats
    fo=pd.DataFrame(fo)
    chiStats = stats.chi2_contingency(observed=fo)
    #critical_value = stats.chi2.ppf(q=1-alpha,df=chiStats[2])
    #observed_chi_val = chiStats[0]
    # p<alpha 等价于 observed_chi_val>critical_value
    chi2_data=(chiStats[1] <= alpha,chiStats[1])
    return chi2_data

def fisher_exact(fo,alpha=0.05):
    '''fisher_exact 显著性检验函数
    此处采用的是调用R的解决方案，需要安装包 pyper
    python解决方案参见
    https://mrnoutahi.com/2016/01/03/Fisher-exac-test-for-mxn-table/
    但还有些问题，所以没用.
    '''
    import pyper as pr
    r=pr.R(use_pandas=True,use_numpy=True)
    r.assign('fo',fo)
    r("b<-fisher.test(fo)")
    pdata=r['b']
    p_value=pdata['p.value']
    if p_value<alpha:
        result=1
    else:
        result=0
    return (result,p_value)

def anova(data,formula):
    '''方差分析
    输入
    --data： DataFrame格式，包含数值型变量和分类型变量
    --formula：变量之间的关系，如：数值型变量~C(分类型变量1)[+C(分类型变量1)[+C(分类型变量1):(分类型变量1)]

    返回[方差分析表]
    [总体的方差来源于组内方差和组间方差，通过比较组间方差和组内方差的比来推断两者的差异]
    --df:自由度
    --sum_sq：误差平方和
    --mean_sq：误差平方和/对应的自由度
    --F：mean_sq之比
    --PR(>F)：p值，比如<0.05则代表有显著性差异
    '''
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    cw_lm=ols(formula, data=data).fit() #Specify C for Categorical
    r=sm.stats.anova_lm(cw_lm)
    return r


def mca(X,N=2):
    '''对应分析函数，暂时支持双因素
    X：观察频数表
    N：返回的维数，默认2维
    可以通过scatter函数绘制：
    fig=scatter([pr,pc])
    fig.savefig('mca.png')
    '''
    from scipy.linalg import diagsvd
    S = X.sum().sum()
    Z = X / S  # correspondence matrix
    r = Z.sum(axis=1)
    c = Z.sum()
    D_r = np.diag(1/np.sqrt(r))
    Z_c = Z - np.outer(r, c)  # standardized residuals matrix
    D_c = np.diag(1/np.sqrt(c))

    # another option, not pursued here, is sklearn.decomposition.TruncatedSVD
    P,s,Q = np.linalg.svd(np.dot(np.dot(D_r, Z_c),D_c))
    #S=diagsvd(s[:2],P.shape[0],2)
    pr=np.dot(np.dot(D_r,P),diagsvd(s[:N],P.shape[0],N))
    pc=np.dot(np.dot(D_c,Q.T),diagsvd(s[:N],Q.shape[0],N))
    inertia=np.cumsum(s**2)/np.sum(s**2)
    inertia=inertia.tolist()
    if isinstance(X,pd.DataFrame):
        pr=pd.DataFrame(pr,index=X.index,columns=list('XYZUVW')[:N])
        pc=pd.DataFrame(pc,index=X.columns,columns=list('XYZUVW')[:N])
    return pr,pc,inertia
    '''
    w=pd.ExcelWriter(u'mca_.xlsx')
    pr.to_excel(w,startrow=0,index_label=True)
    pc.to_excel(w,startrow=len(pr)+2,index_label=True)
    w.save()
    '''

def cluster(data,code,cluster_qq,n_clusters='auto',max_clusters=7):
    '''对态度题进行聚类
    '''

    from sklearn.cluster import KMeans
    #from sklearn.decomposition import PCA
    from sklearn import metrics
    #import prince
    qq_max=sorted(code,key=lambda x:int(re.findall('\d+',x)[0]))[-1]
    new_cluster='Q{}'.format(int(re.findall('\d+',qq_max)[0])+1)
    #new_cluster='Q32'

    qlist=code[cluster_qq]['qlist']
    X=data[qlist]
    # 去除所有态度题选择的分数都一样的用户（含仅有两个不同）
    std_t=min(1.41/np.sqrt(len(qlist)),0.40) if len(qlist)>=8 else 0.10
    X=X[X.T.std()>std_t]
    index_bk=X.index#备份，方便还原
    X.fillna(0,inplace=True)
    X1=X.T
    X1=(X1-X1.mean())/X1.std()
    X1=X1.T.as_matrix()


    if n_clusters == 'auto':
        #聚类个数的选取和评估
        silhouette_score=[]# 轮廊系数
        SSE_score=[]
        klist=np.arange(2,15)
        for k in klist:
            est = KMeans(k)  # 4 clusters
            est.fit(X1)
            tmp=np.sum((X1-est.cluster_centers_[est.labels_])**2)
            SSE_score.append(tmp)
            tmp=metrics.silhouette_score(X1, est.labels_)
            silhouette_score.append(tmp)
        '''
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        fig = plt.figure(2)
        ax.plot(klist,np.array(silhouette_score))
        ax = fig.add_subplot(111)
        ax.plot(klist,np.array(SSE_score))
        '''
        # 找轮廊系数的拐点
        ss=np.array(silhouette_score)
        t1=[False]+list(ss[1:]>ss[:-1])
        t2=list(ss[:-1]>ss[1:])+[False]
        k_log=[t1[i]&t2[i] for i in range(len(t1))]
        if True in k_log:
            k=k_log.index(True)
        else:
            k=1
        k=k if k<=max_clusters-2 else max_clusters-2 # 限制最多分7类
        k_best=klist[k]
    else:
        k_best=n_clusters

    est = KMeans(k_best)  # 4 clusters
    est.fit(X1)

    # 系数计算
    SSE=np.sqrt(np.sum((X1-est.cluster_centers_[est.labels_])**2)/len(X1))
    silhouette_score=metrics.silhouette_score(X1, est.labels_)

    print('有效样本数:{},特征数：{},最佳分类个数：{} 类'.format(len(X1),len(qlist),k_best))
    print('SSE(样本到所在类的质心的距离)为：{:.2f},轮廊系数为: {:.2f}'.format(SSE,silhouette_score))

    # 绘制降维图
    '''
    X_PCA = PCA(2).fit_transform(X1)
    kwargs = dict(cmap = plt.cm.get_cmap('rainbow', 10),
                  edgecolor='none', alpha=0.6)
    labels=pd.Series(est.labels_)
    plt.figure()
    plt.scatter(X_PCA[:, 0], X_PCA[:, 1], c=labels, **kwargs)
    '''

    '''
    # 三维立体图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_PCA[:, 0], X_PCA[:, 1],X_PCA[:, 2], c=labels, **kwargs)
    '''

    # 导出到原数据
    parameters={'methods':'kmeans','inertia':est.inertia_,'SSE':SSE,'silhouette':silhouette_score,\
      'n_clusters':k_best,'n_features':len(qlist),'n_samples':len(X1),'qnum':new_cluster,\
       'data':X1,'labels':est.labels_}
    data[new_cluster]=pd.Series(est.labels_,index=index_bk)
    code[new_cluster]={'content':'态度题聚类结果','qtype':'单选题','qlist':[new_cluster],
        'code':dict(zip(range(k_best),['cluster{}'.format(i+1) for i in range(k_best)]))}
    print('结果已经存进数据, 题号为：{}'.format(new_cluster))
    return data,code,parameters
    '''
    # 对应分析
    t=data.groupby([new_cluster])[code[cluster_qq]['qlist']].mean()
    t.columns=['R{}'.format(i+1) for i in range(len(code[cluster_qq]['qlist']))]
    t=t.rename(index=code[new_cluster]['code'])
    ca=prince.CA(t)
    ca.plot_rows_columns(show_row_labels=True,show_column_labels=True)
    '''



def scatter(data,legend=False,title=None,font_ch=None,find_path=None):
    '''
    绘制带数据标签的散点图
    '''
    import matplotlib.font_manager as fm
    if font_ch is None:
        fontlist=['calibri.ttf','simfang.ttf','simkai.ttf','simhei.ttf','simsun.ttc','msyh.ttf','msyh.ttc']
        myfont=''
        if not find_path:
            find_paths=['C:\\Windows\\Fonts','']
        # fontlist 越靠后越优先，findpath越靠后越优先
        for find_path in find_paths:
            for f in fontlist:
                if os.path.exists(os.path.join(find_path,f)):
                    myfont=os.path.join(find_path,f)
        if len(myfont)==0:
            print('没有找到合适的中文字体绘图，请检查.')
            myfont=None
        else:
            myfont = fm.FontProperties(fname=myfont)
    else:
        myfont=fm.FontProperties(fname=font_ch)
    fig, ax = plt.subplots()
    #ax.grid('on')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.axhline(y=0, linestyle='-', linewidth=1.2, alpha=0.6)
    ax.axvline(x=0, linestyle='-', linewidth=1.2, alpha=0.6)
    color=['blue','red','green','dark']
    if not isinstance(data,list):
        data=[data]
    for i,dd in enumerate(data):
        ax.scatter(dd.iloc[:,0], dd.iloc[:,1], c=color[i], s=50,
                   label=dd.columns[1])
        for _, row in dd.iterrows():
            ax.annotate(row.name, (row.iloc[0], row.iloc[1]), color=color[i],fontproperties=myfont,fontsize=10)
    ax.axis('equal')
    if legend:
        ax.legend(loc='best')
    if title:
        ax.set_title(title,fontproperties=myfont)
    return fig



def sankey(df,filename=None):
    '''SanKey图绘制
    df的列是左节点，行是右节点
    注:暂时没找到好的Python方法，所以只生成R语言所需数据
    返回links 和 nodes
    # R code 参考
    library(networkD3)
    dd=read.csv('price_links.csv')
    links<-data.frame(source=dd$from,target=dd$to,value=dd$value)
    nodes=read.csv('price_nodes.csv',encoding = 'UTF-8')
    nodes<-nodes['name']
    Energy=c(links=links,nodes=nodes)
    sankeyNetwork(Links = links, Nodes = nodes, Source = "source",
                  Target = "target", Value = "value", NodeID = "name",
                  units = "TWh",fontSize = 20,fontFamily='微软雅黑',nodeWidth=20)
    '''
    nodes=['Total']
    nodes=nodes+list(df.columns)+list(df.index)
    nodes=pd.DataFrame(nodes)
    nodes['id']=range(len(nodes))
    nodes.columns=['name','id']
    R,C=df.shape
    t1=pd.DataFrame(df.as_matrix(),columns=range(1,C+1),index=range(C+1,R+C+1))
    t1.index.name='to'
    t1.columns.name='from'
    links=t1.unstack().reset_index(name='value')
    links0=pd.DataFrame({'from':[0]*C,'to':range(1,C+1),'value':list(df.sum())})
    links=links0.append(links)
    if filename:
        links.to_csv(filename+'_links.csv',index=False,encoding='utf-8')
        nodes.to_csv(filename+'_nodes.csv',index=False,encoding='utf-8')
    return (links,nodes)


def table(data,code,total=True):
    '''
    单个题目描述统计
    code是data的编码，列数大于1
    返回字典格式数据：
    'fop'：百分比, 对于单选题和为1，多选题分母为样本数
    'fo'： 观察频数表，其中添加了合计项
    'fw':  加权频数表，可实现平均值、T2B等功能，仅当code中存在关键词'weight'时才有
    '''
    # 单选题
    qtype=code['qtype']
    index=code['qlist']
    data=pd.DataFrame(data)
    sample_len=data[code['qlist']].notnull().T.any().sum()
    result={}
    if qtype == u'单选题':
        fo=data.iloc[:,0].value_counts()
        if 'weight' in code:
            w=pd.Series(code['weight'])
            fo1=fo[w.index][fo[w.index].notnull()]
            fw=(fo1*w).sum()/fo1.sum()
            result['fw']=fw
        fo.sort_values(ascending=False,inplace=True)
        fop=fo.copy()
        fop=fop/fop.sum()*1.0
        fop[u'合计']=fop.sum()
        fo[u'合计']=fo.sum()
        if 'code' in code:
            fop.rename(index=code['code'],inplace=True)
            fo.rename(index=code['code'],inplace=True)
        fop.name=u'占比'
        fo.name=u'频数'
        fop=pd.DataFrame(fop)
        fo=pd.DataFrame(fo)
        result['fo']=fo
        result['fop']=fop
    elif qtype == u'多选题':
        fo=data.sum()
        fo.sort_values(ascending=False,inplace=True)
        fo[u'合计']=fo.sum()
        if 'code' in code:
            fo.rename(index=code['code'],inplace=True)
        fop=fo.copy()
        fop=fop/sample_len
        fop.name=u'占比'
        fo.name=u'频数'
        fop=pd.DataFrame(fop)
        fo=pd.DataFrame(fo)
        result['fop']=fop
        result['fo']=fo
    elif qtype == u'矩阵单选题':
        fo=pd.DataFrame(columns=code['qlist'],index=sorted(code['code']))
        for i in fo.columns:
            fo.loc[:,i]=data[i].value_counts()
        if 'weight' not in code:
            code['weight']=dict(zip(code['code'].keys(),code['code'].keys()))
        fw=pd.DataFrame(columns=[u'加权'],index=code['qlist'])
        w=pd.Series(code['weight'])
        for c in fo.columns:
            t=fo[c]
            t=t[w.index][t[w.index].notnull()]
            if t.sum()>1e-17:
                fw.loc[c,u'加权']=(t*w).sum()/t.sum()
            else:
                fw.loc[c,u'加权']=0
        fw.rename(index=code['code_r'],inplace=True)
        result['fw']=fw
        result['weight']=','.join(['{}:{}'.format(code['code'][c],code['weight'][c]) for c in code['code']])
        fo.rename(columns=code['code_r'],index=code['code'],inplace=True)
        fop=fo.copy()
        fop=fop/sample_len
        result['fop']=fop
        result['fo']=fo
    elif qtype == u'排序题':
        #提供综合统计和TOP1值统计
        # 其中综合的算法是当成单选题，给每个TOP分配和为1的权重
        #topn=max([len(data[q][data[q].notnull()].unique()) for q in index])
        #topn=len(index)
        topn=data[index].fillna(0).max().max()
        topn=int(topn)
        qsort=dict(zip([i+1 for i in range(topn)],[(topn-i)*2.0/(topn+1)/topn for i in range(topn)]))
        top1=data.applymap(lambda x:int(x==1))
        data_weight=data.replace(qsort)
        t1=pd.DataFrame()
        t1['TOP1']=top1.sum()
        t1[u'综合']=data_weight.sum()
        t1.sort_values(by=u'综合',ascending=False,inplace=True)
        t1.rename(index=code['code'],inplace=True)
        t=t1.copy()
        t=t/sample_len
        result['fop']=t
        result['fo']=t1
        # 新增topn矩阵
        t_topn=pd.DataFrame()
        for i in range(topn):
            t_topn['TOP%d'%(i+1)]=data.applymap(lambda x:int(x==i+1)).sum()
        t_topn.sort_values(by=u'TOP1',ascending=False,inplace=True)
        if 'code' in code:
            t_topn.rename(index=code['code'],inplace=True)
        result['TOPN_fo']=t_topn#频数
        result['TOPN']=t_topn/sample_len
        result['weight']='+'.join(['TOP{}*{:.2f}'.format(i+1,(topn-i)*2.0/(topn+1)/topn) for i in range(topn)])
    else:
        result['fop']=None
        result['fo']=None
    if (not total) and not(result['fo'] is None) and (u'合计' in result['fo'].index):
        result['fo'].drop([u'合计'],axis=0,inplace=True)
        result['fop'].drop([u'合计'],axis=0,inplace=True)
    if not(result['fo'] is None) and ('code_order' in code):
        code_order=[q for q in code['code_order'] if q in result['fo'].index]
        if u'合计' in result['fo'].index:
            code_order=code_order+[u'合计']
        result['fo']=pd.DataFrame(result['fo'],index=code_order)
        result['fop']=pd.DataFrame(result['fop'],index=code_order)
    return result

def crosstab(data_index,data_column,code_index=None,code_column=None,qtype=None,total=True):
    '''适用于问卷数据的交叉统计
    输入参数：
    data_index: 因变量，放在行中
    data_column:自变量，放在列中
    code_index: dict格式，指定data_index的编码等信息
    code_column: dict格式，指定data_column的编码等信息
    qtype: 给定两个数据的题目类型，若为字符串则给定data_index，若为列表，则给定两个的
    返回字典格式数据
    'fop'：默认的百分比表，行是data_index,列是data_column
    'fo'：原始频数表，且添加了总体项
    'fw': 加权平均值

    简要说明：
    因为要处理各类题型，这里将单选题处理为多选题

    fo：观察频数表
    nij是同时选择了Ri和Cj的频数
    总体的频数是选择了Ri的频数，与所在行的总和无关
    行变量\列变量  C1 |C2 | C3| C4|总体
             R1|  n11|n12|n13|n14|n1:
             R2|  n21|n22|n23|n23|n2:
             R3|  n31|n32|n33|n34|n3:
     fop: 观察百分比表(列变量)
     这里比较难处理，data_column各个类别的样本量和总体的样本量不一样，各类别的样本量为同时
     选择了行变量和列类别的频数。而总体的样本量为选择了行变量的频数
     fw: 加权平均值
     如果data_index的编码code含有weight字段，则我们会输出分组的加权平均值


    '''

    # 将Series转为DataFrame格式
    data_index=pd.DataFrame(data_index)
    data_column=pd.DataFrame(data_column)

    # 获取行/列变量的题目类型
    #  默认值
    if data_index.shape[1]==1:
        qtype1=u'单选题'
    else:
        qtype1=u'多选题'
    if data_column.shape[1]==1:
        qtype2=u'单选题'
    else:
        qtype2=u'多选题'
    #  根据参数修正
    if code_index:
        qtype1=code_index['qtype']
        if qtype1 == u'单选题':
            data_index.replace(code_index['code'],inplace=True)
        elif qtype1 in [u'多选题',u'排序题']:
            data_index.rename(columns=code_index['code'],inplace=True)
        elif qtype1 == u'矩阵单选题':
            data_index.rename(columns=code_index['code_r'],inplace=True)
    if code_column:
        qtype2=code_column['qtype']
        if qtype2 == u'单选题':
            data_column.replace(code_column['code'],inplace=True)
        elif qtype2 in [u'多选题',u'排序题']:
            data_column.rename(columns=code_column['code'],inplace=True)
        elif qtype2 == u'矩阵单选题':
            data_column.rename(columns=code_column['code_r'],inplace=True)
    if qtype:
        #qtype=list(qtype)
        if isinstance(qtype,list) and len(qtype)==2:
            qtype1=qtype[0]
            qtype2=qtype[1]
        elif isinstance(qtype,str):
            qtype1=qtype
    if qtype1 == u'单选题':
        data_index=sa_to_ma(data_index)
        qtype1=u'多选题'
    # 将单选题变为多选题
    if qtype2 == u'单选题':
        #data_column=pd.get_dummies(data_column.iloc[:,0])
        data_column=sa_to_ma(data_column)
        qtype2=u'多选题'

    # 准备工作
    index_list=list(data_index.columns)
    columns_list=list(data_column.columns)
    # 频数表/data_column各个类别的样本量
    column_freq=data_column.iloc[list(data_index.notnull().T.any()),:].sum()
    #column_freq[u'总体']=column_freq.sum()
    column_freq[u'总体']=data_index.notnull().T.any().sum()
    R=len(index_list)
    C=len(columns_list)
    result={}
    result['sample_size']=column_freq
    if (qtype1 == u'多选题') and (qtype2 == u'多选题'):
        data_index.fillna(0,inplace=True)
        t=pd.DataFrame(np.dot(data_index.fillna(0).T,data_column.fillna(0)))
        t.rename(index=dict(zip(range(R),index_list)),columns=dict(zip(range(C),columns_list)),inplace=True)
        if code_index and ('weight' in code_index):
            w=pd.Series(code_index['weight'])
            w.rename(index=code_index['code'],inplace=True)
            fw=pd.DataFrame(columns=[u'加权'],index=t.columns)
            for c in t.columns:
                tmp=t[c]
                tmp=tmp[w.index][tmp[w.index].notnull()]
                if abs(tmp.sum())>0:
                    fw.loc[c,u'加权']=(tmp*w).sum()/tmp.sum()
                else:
                    fw.loc[c,u'加权']=0
            fo1=data_index.sum()[w.index][data_index.sum()[w.index].notnull()]
            if abs(fo1.sum())>0:
                fw.loc[u'总体',u'加权']=(fo1*w).sum()/fo1.sum()
            else:
                fw.loc[u'总体',u'加权']=0
            result['fw']=fw
        t[u'总体']=data_index.sum()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        for i in t.columns:
            if column_freq[i]!=0:
                t.loc[:,i]=t.loc[:,i]/column_freq[i]
        result['fop']=t
        result['fo']=t1
    elif (qtype1 == u'矩阵单选题') and (qtype2 == u'多选题'):
        if code_index and ('weight' in code_index):
            data_index.replace(code_index['weight'],inplace=True)
        t=pd.DataFrame(np.dot(data_index.fillna(0).T,data_column.fillna(0)))
        t=pd.DataFrame(np.dot(t,np.diag(1/data_column.sum())))
        t.rename(index=dict(zip(range(R),index_list)),columns=dict(zip(range(C),columns_list)),inplace=True)
        t[u'总体']=data_index.mean()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        result['fop']=t
        result['fo']=t1
    elif (qtype1 == u'排序题') and (qtype2 == u'多选题'):
        topn=int(data_index.max().max())
        #topn=max([len(data_index[q][data_index[q].notnull()].unique()) for q in index_list])
        qsort=dict(zip([i+1 for i in range(topn)],[(topn-i)*2.0/(topn+1)/topn for i in range(topn)]))
        data_index_zh=data_index.replace(qsort)
        t=pd.DataFrame(np.dot(data_index_zh.fillna(0).T,data_column.fillna(0)))
        t.rename(index=dict(zip(range(R),index_list)),columns=dict(zip(range(C),columns_list)),inplace=True)
        t[u'总体']=data_index_zh.sum()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        for i in t.columns:
            if column_freq[i]!=0:
                t.loc[:,i]=t.loc[:,i]/column_freq[i]
        result['fop']=t
        result['fo']=t1
        # 新增TOP1 数据
        data_index_top1=data_index.applymap(lambda x:int(x==1))
        top1=pd.DataFrame(np.dot(data_index_top1.fillna(0).T,data_column.fillna(0)))
        top1.rename(index=dict(zip(range(R),index_list)),columns=dict(zip(range(C),columns_list)),inplace=True)
        top1[u'总体']=data_index_top1.fillna(0).sum()
        top1.sort_values([u'总体'],ascending=False,inplace=True)
        for i in top1.columns:
            if column_freq[i]!=0:
                top1.loc[:,i]=top1.loc[:,i]/column_freq[i]
        result['TOP1']=top1
    else:
        result['fop']=None
        result['fo']=None
    # 去除总体
    if (not total) and not(result['fo'] is None) and ('总体' in result['fo'].columns):
        result['fo'].drop(['总体'],axis=1,inplace=True)
        result['fop'].drop(['总体'],axis=1,inplace=True)
    # 顺序重排
    if not(result['fo'] is None) and code_index and ('code_order' in code_index) and qtype1!='矩阵单选题':
        code_order=code_index['code_order']
        code_order=[q for q in code_order if q in result['fo'].index]
        if u'总体' in result['fo'].index:
            code_order=code_order+[u'总体']
        result['fo']=pd.DataFrame(result['fo'],index=code_order)
        result['fop']=pd.DataFrame(result['fop'],index=code_order)
    if not(result['fo'] is None) and code_column and ('code_order' in code_column) and qtype2!='矩阵单选题':
        code_order=code_column['code_order']
        code_order=[q for q in code_order if q in result['fo'].columns]
        if u'总体' in result['fo'].columns:
            code_order=code_order+[u'总体']
        result['fo']=pd.DataFrame(result['fo'],columns=code_order)
        result['fop']=pd.DataFrame(result['fop'],columns=code_order)
    return result



def ncrosstab(data_index,data_column,code_index=None,code_column=None,qtype=None):
    '''适用于问卷数据的交叉统计【后期将删除】
    输入参数：
    data_index: 因变量，放在行中
    data_column:自变量，放在列中
    qtype: 给定两个数据的题目类型，若为字符串则给定data_index，若为列表，则给定两个的
    code_index: dict格式，指定data_index的编码等信息
    code_column: dict格式，指定data_column的编码等信息
    返回数据：(t,t1)
    t：默认的百分比表，行是data_index,列是data_column
    t1：原始频数表，且添加了总体项
    '''

    # 将Series转为DataFrame格式
    data_index=pd.DataFrame(data_index)
    data_column=pd.DataFrame(data_column)

    # 获取行/列变量的题目类型
    #  默认值
    if data_index.shape[1]==1:
        qtype1=u'单选题'
    else:
        qtype1=u'多选题'
    if data_column.shape[1]==1:
        qtype2=u'单选题'
    else:
        qtype2=u'多选题'
    #  根据参数修正
    if code_index:
        qtype1=code_index['qtype']
        if qtype1 == u'单选题':
            data_index.replace(code_index['code'],inplace=True)
        elif qtype1 in [u'多选题',u'排序题']:
            data_index.rename(columns=code_index['code'],inplace=True)
        elif qtype1 == u'矩阵单选题':
            data_index.rename(columns=code_index['code_r'],inplace=True)
    if code_column:
        qtype2=code_column['qtype']
        if qtype2 == u'单选题':
            data_column.replace(code_column['code'],inplace=True)
        elif qtype2 in [u'多选题',u'排序题']:
            data_column.rename(columns=code_column['code'],inplace=True)
        elif qtype2 == u'矩阵单选题':
            data_column.rename(columns=code_column['code_r'],inplace=True)
    if qtype:
        qtype=list(qtype)
        if len(qtype)==2:
            qtype1=qtype[0]
            qtype2=qtype[1]
        else:
            qtype1=qtype[0]

    if qtype1 == u'单选题':
        data_index=sa_to_ma(data_index)
        qtype1=u'多选题'
    # 将单选题变为多选题
    if qtype2 == u'单选题':
        data_column=sa_to_ma(data_column)
        qtype2=u'多选题'

    # 准备工作
    index_list=list(data_index.columns)
    columns_list=list(data_column.columns)
    # 次数频数表为同时选择
    column_freq=data_column.iloc[list(data_index.notnull().T.any()),:].sum()
    #column_freq=data_column.sum()
    column_freq[u'总体']=column_freq.sum()
    R=len(index_list)
    C=len(columns_list)

    if (qtype1 == u'多选题') and (qtype2 == u'多选题'):
        data_index.fillna(0,inplace=True)
        t=pd.DataFrame(np.dot(data_index.fillna(0).T,data_column.fillna(0)))
        t.rename(index=dict(zip(range(R),index_list)),columns=dict(zip(range(C),columns_list)),inplace=True)
        t[u'总体']=data_index.sum()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        for i in t.columns:
            t.loc[:,i]=t.loc[:,i]/column_freq[i]

    elif (qtype1 == u'矩阵单选题') and (qtype2 == u'多选题'):
        t=pd.DataFrame(np.dot(data_index.fillna(0).T,data_column.fillna(0)))
        t=pd.DataFrame(np.dot(t,np.diag(1/data_column.sum())))
        t.rename(index=dict(zip(range(R),index_list)),columns=dict(zip(range(C),columns_list)),inplace=True)
        t[u'总体']=data_index.mean()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()


    elif (qtype1 == u'排序题') and (qtype2 == u'多选题'):
        topn=int(data_index.max().max())
        #topn=max([len(data_index[q][data_index[q].notnull()].unique()) for q in index_list])
        qsort=dict(zip([i+1 for i in range(topn)],[(topn-i)*2.0/(topn+1)/topn for i in range(topn)]))
        data_index.replace(qsort,inplace=True)
        t=pd.DataFrame(np.dot(data_index.fillna(0).T,data_column.fillna(0)))
        t.rename(index=dict(zip(range(R),index_list)),columns=dict(zip(range(C),columns_list)),inplace=True)
        t[u'总体']=data_index.sum()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        for i in t.columns:
            t.loc[:,i]=t.loc[:,i]/column_freq[i]
    else:
        t=None
        t1=None
    return (t,t1)



def qtable(data,*args,**kwargs):
    '''简易频数统计函数
    输入
    data：数据框，可以是所有的数据
    code:数据编码
    q1: 题目序号
    q2: 题目序号
    # 单个变量的频数统计
    qtable(data,code,'Q1')
    # 两个变量的交叉统计
    qtable(data,code,'Q1','Q2')

    '''
    code=None
    q1=None
    q2=None
    for a in args:
        if (isinstance(a,str)) and (not q1):
            q1=a
        elif (isinstance(a,str)) and (q1):
            q2=a
        elif isinstance(a,dict):
            code=a
    if not code:
        code=data_auto_code(data)
    if not q1:
        print('please input the q1,such as Q1.')
        return
    total=False
    for key in kwargs:
        if key == 'total':
            total=kwargs['total']
    if q2 is None:
        result=table(data[code[q1]['qlist']],code[q1],total=total)
    else:
        result=crosstab(data[code[q1]['qlist']],data[code[q2]['qlist']],code[q1],code[q2],total=total)
    return result

def association_rules(df,minSup=0.08,minConf=0.4,Y=None):
    '''关联规则分析
    df是一个观察频数表，返回其中存在的关联规则

    '''
    try :
        import relations as rlt
    except :
        print('没有找到关联分析需要的包: import relations')
        return (None,None,None)
    a=rlt.apriori(df, minSup, minConf)
    rules,freq=a.genRules(Y=Y)
    if rules is None:
        return (None,None,None)
    result=';\n'.join(['{}:  支持度={:.1f}%, 置信度={:.1f}%'.format(rules.loc[ii,'rule'],100*rules.loc[ii,'sup'],100*rules.loc[ii,'conf']) for ii in rules.index[:4]])
    return (result,rules,freq)



def contingency(fo,alpha=0.05):
    ''' 列联表分析：(观察频数表分析)
    # 预增加一个各类别之间的距离
    1、生成TGI指数、TWI指数、CHI指数
    2、独立性检验
    3、当两个变量不显著时，考虑单个之间的显著性
    返回字典格式
    chi_test: 卡方检验结果，1:显著；0:不显著；-1：期望值不满足条件
    coef: 包含chi2、p值、V相关系数
    log: 记录一些异常情况
    FO: 观察频数
    FE: 期望频数
    TGI：fo/fe
    TWI：fo-fe
    CHI：sqrt((fo-fe)(fo/fe-1))*sign(fo-fe)
    significant:{
    .'result': 显著性结果[1(显著),0(不显著),-1(fe小于5的过多)]
    .'pvalue':
    .'method': chi_test or fisher_test
    .'vcoef':
    .'threshold':
    }
    summary:{
    .'summary': 结论提取
    .'fit_test': 拟合优度检验
    .'chi_std':
    .'chi_mean':
    '''
    import scipy.stats as stats
    cdata={}
    if isinstance(fo,pd.core.series.Series):
        fo=pd.DataFrame(fo)
    if not isinstance(fo,pd.core.frame.DataFrame):
        return cdata
    R,C=fo.shape
    # 去除所有的总体、合计、其他、其它
    if u'总体' in fo.columns:
        fo.drop([u'总体'],axis=1,inplace=True)
    if any([(u'其他' in '%s'%s) or (u'其它' in '%s'%s) for s in fo.columns]):
        tmp=[s for s in fo.columns if (u'其他' in s) or (u'其它' in s)]
        for t in tmp:
            fo.drop([t],axis=1,inplace=True)
    if u'合计' in fo.index:
        fo.drop([u'合计'],axis=0,inplace=True)
    if any([(u'其他' in '%s'%s) or (u'其它' in '%s'%s) for s in fo.index]):
        tmp=[s for s in fo.index if (u'其他' in s) or (u'其它' in s)]
        for t in tmp:
            fo.drop([t],axis=0,inplace=True)
    fe=fo.copy()
    N=fo.sum().sum()
    if N==0:
        #print('rpt.contingency:: fo的样本数为0,请检查数据')
        return cdata
    for i in fe.index:
        for j in fe.columns:
            fe.loc[i,j]=fe.loc[i,:].sum()*fe.loc[:,j].sum()/float(N)
    TGI=fo/fe
    TWI=fo-fe
    CHI=np.sqrt((fo-fe)**2/fe)*(TWI.applymap(lambda x: int(x>0))*2-1)
    PCHI=1/(1+np.exp(-1*CHI))
    cdata['FO']=fo
    cdata['FE']=fe
    cdata['TGI']=TGI*100
    cdata['TWI']=TWI
    cdata['CHI']=CHI
    cdata['PCHI']=PCHI

    # 显著性检验(独立性检验)
    significant={}
    significant['threshold']=stats.chi2.ppf(q=1-alpha,df=C-1)
    #threshold=math.ceil(R*C*0.2)# 期望频数和实际频数不得小于5

    # 去除行、列变量中样本数和过低的变量
    threshold=max(3,min(30,N*0.05))
    ind1=fo.sum(axis=1)>=threshold
    ind2=fo.sum()>=threshold
    fo=fo.loc[ind1,ind2]

    if (fo.shape[0]<=1) or (np.any(fo.sum()==0)) or (np.any(fo.sum(axis=1)==0)):
        significant['result']=-2
        significant['pvalue']=-2
        significant['method']='fo not frequency'
    #elif ((fo<=5).sum().sum()>=threshold):
        #significant['result']=-1
        #significant['method']='need fisher_exact'
        '''fisher_exact运行所需时间极其的长，此处还是不作检验
        fisher_r,fisher_p=fisher_exact(fo)
        significant['pvalue']=fisher_p
        significant['method']='fisher_exact'
        significant['result']=fisher_r
        '''
    else:
        try:
            chiStats = stats.chi2_contingency(observed=fo)
        except:
            chiStats=(1,np.nan)
        significant['pvalue']=chiStats[1]
        significant['method']='chi-test'
        #significant['vcoef']=math.sqrt(chiStats[0]/N/min(R-1,C-1))
        if chiStats[1] <= alpha:
            significant['result']=1
        elif np.isnan(chiStats[1]):
            significant['pvalue']=-2
            significant['result']=-1
        else:
            significant['result']=0
    cdata['significant']=significant

    # 列联表分析summary
    chi_sum=(CHI**2).sum(axis=1)
    chi_value_fit=stats.chi2.ppf(q=1-alpha,df=C-1)#拟合优度检验
    fit_test=chi_sum.map(lambda x : int(x>chi_value_fit))
    summary={}
    summary['fit_test']=fit_test
    summary['chi_std']=CHI.unstack().std()
    summary['chi_mean']=CHI.unstack().mean()
    #print('the std of CHI is %.2f'%summary['chi_std'])
    conclusion=''
    fo_rank=fo.sum().rank(ascending=False)# 给列选项排名，只分析排名在前4选项的差异
    for c in fo_rank[fo_rank<5].index:#CHI.columns:
        #针对每一列，选出大于一倍方差的行选项，如果过多，则只保留前三个
        tmp=list(CHI.loc[CHI[c]-summary['chi_mean']>summary['chi_std'],c].sort_values(ascending=False)[:3].index)
        tmp=['%s'%s for s in tmp]# 把全部内容转化成字符串
        if tmp:
            tmp1=u'{col}：{s}'.format(col=c,s=' || '.join(tmp))
            conclusion=conclusion+tmp1+'; \n'
    if significant['result']==1:
        if conclusion:
            tmp='在95%置信水平下显著性检验(卡方检验)结果为*显著*, 且CHI指标在一个标准差外的(即相对有差异的)有：\n'
        else:
            tmp='在95%置信水平下显著性检验(卡方检验)结果为*显著*，但没有找到相对有差异的配对'
    elif significant['result']==0:
        if conclusion:
            tmp='在95%置信水平下显著性检验(卡方检验)结果为*不显著*, 但CHI指标在一个标准差外的(即相对有差异的)有：\n'
        else:
            tmp='在95%置信水平下显著性检验(卡方检验)结果为*不显著*，且没有找到相对有差异的配对'
    else:
        if conclusion:
            tmp='不满足显著性检验(卡方检验)条件, 但CHI指标在一个标准差外的(即相对有差异的)有：\n'
        else:
            tmp='不满足显著性检验(卡方检验)条件，且没有找到相对有差异的配对'
    conclusion=tmp+conclusion

    summary['summary']=conclusion
    cdata['summary']=summary
    return cdata

def pre_cross_qlist(data,code):
    '''自适应给出可以进行交叉分析的变量和相应选项
    满足以下条件的将一键交叉分析：
    1、单选题
    2、如果选项是文本，则平均长度应小于10
    ...
    返回：
    cross_qlist: [[题目序号,变量选项],]
    '''
    cross_qlist=[]
    for qq in code:
        qtype=code[qq]['qtype']
        qlist=code[qq]['qlist']
        content=code[qq]['content']
        sample_len_qq=data[code[qq]['qlist']].notnull().T.any().sum()
        if qtype not in ['单选题']:
            continue
        if not(set(qlist) <= set(data.columns)):
            continue
        t=qtable(data,code,qq)['fo']
        if 'code_order' in code[qq]:
            code_order=code[qq]['code_order']
            code_order=[q for q in code_order if q in t.index]
            t=pd.DataFrame(t,index=code_order)
        items=list(t.index)
        code_values=list(code[qq]['code'].values())
        if len(items)<=1:
            continue
        if all([isinstance(t,str) for t in code_values]):
            if sum([len(t) for t in code_values])/len(code_values)>15:
                continue
        if ('code_order' in code[qq]) and (len(items)<10):
            code_order=[q for q in code[qq]['code_order'] if q in t.index]
            t=pd.DataFrame(t,index=code_order)
            ind=np.where(t['频数']>=10)[0]
            if len(ind)>0:
                cross_order=list(t.index[range(ind[0],ind[-1]+1)])
                cross_qlist.append([qq,cross_order])
            continue
        if re.findall('性别|年龄|gender|age',content.lower()):
            cross_qlist.append([qq,items])
            continue
        if (len(items)<=sample_len_qq/30) and (len(items)<10):
            cross_order=list(t.index[t['频数']>=10])
            if cross_order:
                cross_qlist.append([qq,cross_order])
            continue
    return cross_qlist

'''
import report as rpt
ppt=rpt.Report(template)
ppt.add_cover(filename)
ppt.add_slide(data=,title)
ppt.save()
ppt.plo

'''



def cross_chart(data,code,cross_class,filename=u'交叉分析报告', cross_qlist=None,\
delclass=None,plt_dstyle=None,cross_order=None,reverse_display=False,\
total_display=True,max_column_chart=20,save_dstyle=None,template=None):

    '''使用帮助
    data: 问卷数据，包含交叉变量和所有的因变量
    code: 数据编码
    cross_class: 交叉变量，单选题或者多选题，例如：Q1
    filename：文件名,用于PPT和保存相关数据
    cross_list: 需要进行交叉分析的变量，缺省为code中的所有变量
    delclass: 交叉变量中需要删除的单个变量，缺省空
    plt_dstyle: 绘制图表需要用的数据类型，默认为百分比表，可以选择['TGI'、'CHI'、'TWI']等
    save_dstyle: 需要保存的数据类型，格式为列表。
    cross_order: 交叉变量中各个类别的顺序，可以缺少
    total_display: PPT绘制图表中是否显示总体情况
    max_column_chart: 列联表的列数，小于则用柱状图，大于则用条形图
    template: PPT模板信息，{'path': 'layouts':}缺省用自带的。
    '''
    # ===================参数预处理=======================
    if plt_dstyle:
        plt_dstyle=plt_dstyle.upper()

    if not cross_qlist:
        try:
            cross_qlist=list(sorted(code,key=lambda c: int(re.findall('\d+',c)[0])))
        except:
            cross_qlist=list(code.keys())
    if cross_class in cross_qlist:
        cross_qlist.remove(cross_class)

    # =================基本数据获取==========================
    #交叉分析的样本数统一为交叉变量的样本数
    sample_len=data[code[cross_class]['qlist']].notnull().T.any().sum()


    # 交叉变量中每个类别的频数分布.
    if code[cross_class]['qtype'] == u'单选题':
        #data[cross_class].replace(code[cross_class]['code'],inplace=True)
        cross_class_freq=data[code[cross_class]['qlist'][0]].value_counts()
        cross_class_freq[u'合计']=cross_class_freq.sum()
        cross_class_freq.rename(index=code[cross_class]['code'],inplace=True)
        #cross_columns_qlist=code[cross_class]['qlist']
    elif code[cross_class]['qtype'] == u'多选题':
        cross_class_freq=data[code[cross_class]['qlist']].sum()
        cross_class_freq[u'合计']=cross_class_freq.sum()
        cross_class_freq.rename(index=code[cross_class]['code'],inplace=True)
        #data.rename(columns=code[cross_class]['code'],inplace=True)
        #cross_columns_qlist=[code[cross_class]['code'][k] for k in code[cross_class]['qlist']]
    elif code[cross_class]['qtype'] == u'排序题':
        tmp=qtable(data,code,cross_class)
        #tmp,tmp1=table(data[code[cross_class]['qlist']],code[cross_class])
        cross_class_freq=tmp['fo'][u'综合']
        cross_class_freq[u'合计']=cross_class_freq.sum()

    # ================I/O接口=============================
    # pptx 接口
    prs=rpt.Report(template) if template else rpt.Report()

    if not os.path.exists('.\\out'):
        os.mkdir('.\\out')
    # 生成数据接口(因为exec&eval)
    Writer=pd.ExcelWriter('.\\out\\'+filename+u'.xlsx')
    Writer_save={}
    if save_dstyle:
        for dstyle in save_dstyle:
            Writer_save[u'Writer_'+dstyle]=pd.ExcelWriter('.\\out\\'+filename+u'_'+dstyle+'.xlsx')

    result={}#记录每道题的的统计数据，用户函数的返回数据

    # 记录没到题目的样本数和显著性差异检验结果，用于最后的数据输出
    cross_columns=list(cross_class_freq.index)
    cross_columns=[r for r in cross_columns if r!=u'合计']
    cross_columns=['内容','题型']+cross_columns+[u'总体',u'显著性检验']
    conclusion=pd.DataFrame(index=cross_qlist,columns=cross_columns)
    conclusion.to_excel(Writer,u'索引')

    # ================封面页=============================
    prs.add_cover(title=filename)

    # ================背景页=============================
    title=u'说明'
    summary=u'交叉题目为'+cross_class+u': '+code[cross_class]['content']
    summary=summary+'\n'+u'各类别样本量如下：'
    prs.add_slide(data={'data':cross_class_freq,'slide_type':'table'},title=title,\
                  summary=summary)

    data_column=data[code[cross_class]['qlist']]
    for qq in cross_qlist:
        # 遍历所有题目
        #print(qq)
        qtitle=code[qq]['content']
        qlist=code[qq]['qlist']
        qtype=code[qq]['qtype']
        if not(set(qlist) <= set(data.columns)):
            continue
        data_index=data[qlist]

        sample_len=data_column.iloc[list(data_index.notnull().T.any()),:].notnull().T.any().sum()
        summary=None
        if qtype not in [u'单选题',u'多选题',u'排序题',u'矩阵单选题']:
            continue
        # 交叉统计
        try:
            if reverse_display:
                result_t=crosstab(data_column,data_index,code_index=code[cross_class],code_column=code[qq])
            else:
                result_t=crosstab(data_index,data_column,code_index=code[qq],code_column=code[cross_class])
        except :
            print('脚本在处理{}时出了一天小问题.....')
            continue
        if ('fo' in result_t) and ('fop' in result_t):
            t=result_t['fop']
            t1=result_t['fo']
            qsample=result_t['sample_size']
        else:
            continue

        if t is None:
            continue

        # =======数据修正==============
        if cross_order and (not reverse_display):
            if u'总体' not in cross_order:
                cross_order=cross_order+[u'总体']
            cross_order=[q for q in cross_order if q in t.columns]
            t=pd.DataFrame(t,columns=cross_order)
            t1=pd.DataFrame(t1,columns=cross_order)
        if cross_order and reverse_display:
            cross_order=[q for q in cross_order if q in t.index]
            t=pd.DataFrame(t,index=cross_order)
            t1=pd.DataFrame(t1,index=cross_order)
        '''在crosstab中已经重排了
        if 'code_order' in code[qq] and qtype!='矩阵单选题':
            code_order=code[qq]['code_order']
            if reverse_display:
                #code_order=[q for q in code_order if q in t.columns]
                if u'总体' in t1.columns:
                    code_order=code_order+[u'总体']
                t=pd.DataFrame(t,columns=code_order)
                t1=pd.DataFrame(t1,columns=code_order)
            else:
                #code_order=[q for q in code_order if q in t.index]
                t=pd.DataFrame(t,index=code_order)
                t1=pd.DataFrame(t1,index=code_order)
        '''
        t.fillna(0,inplace=True)
        t1.fillna(0,inplace=True)

        # =======保存到Excel中========
        t2=pd.concat([t,t1],axis=1)
        t2.to_excel(Writer,qq,index_label=qq,float_format='%.3f')
        Writer_rows=len(t2)# 记录当前Excel文件写入的行数
        pd.DataFrame(qsample,columns=['样本数']).to_excel(Writer,qq,startrow=Writer_rows+2)
        Writer_rows+=len(qsample)+2

        #列联表分析
        cdata=contingency(t1,alpha=0.05)# 修改容错率
        result[qq]=cdata
        if cdata:
            summary=cdata['summary']['summary']
            # 保存各个指标的数据
            if save_dstyle:
                for dstyle in save_dstyle:
                    cdata[dstyle].to_excel(Writer_save[u'Writer_'+dstyle],qq,index_label=qq,float_format='%.2f')

        if qtype in [u'单选题',u'多选题',u'排序题']:
            plt_data=t*100
        else:
            plt_data=t.copy()
        if (abs(1-plt_data.sum())<=0.01+1e-17).all():
            plt_data=plt_data*100


        # ========================【特殊题型处理区】================================
        if 'fw' in result_t:
            plt_data=result_t['fw']
            if cross_order and (not reverse_display):
                if u'总体' not in cross_order:
                    cross_order=cross_order+[u'总体']
                cross_order=[q for q in cross_order if q in plt_data.index]
                plt_data=pd.DataFrame(plt_data,index=cross_order)
            plt_data.to_excel(Writer,qq,startrow=Writer_rows+2)
            Writer_rows+=len(plt_data)

        if plt_dstyle and isinstance(cdata,dict) and (plt_dstyle in cdata):
            plt_data=cdata[plt_dstyle]

        # 绘制PPT
        title=qq+'['+qtype+']: '+qtitle
        if not summary:
            summary=u'这里是结论区域.'
        if 'significant' in cdata:
            sing_result=cdata['significant']['result']
            sing_pvalue=cdata['significant']['pvalue']
        else:
            sing_result=-2
            sing_pvalue=-2
        footnote=u'显著性检验的p值为{:.3f},数据来源于{},样本N={}'.format(sing_pvalue,qq,sample_len)

        # 保存相关数据
        conclusion.loc[qq,:]=qsample
        conclusion.loc[qq,[u'内容',u'题型']]=pd.Series({u'内容':code[qq]['content'],u'题型':code[qq]['qtype']})
        conclusion.loc[qq,u'显著性检验']=sing_result

        if (not total_display) and (u'总体' in plt_data.columns):
            plt_data.drop([u'总体'],axis=1,inplace=True)

        if len(plt_data)>max_column_chart:
            prs.add_slide(data={'data':plt_data[::-1],'slide_type':'chart','type':'BAR_CLUSTERED'},\
                          title=title,summary=summary,footnote=footnote)
        else:
            prs.add_slide(data={'data':plt_data,'slide_type':'chart','type':'COLUMN_CLUSTERED'},\
                          title=title,summary=summary,footnote=footnote)
        # 排序题特殊处理
        if (qtype == u'排序题') and ('TOP1' in result_t):
            plt_data=result_t['TOP1']*100
            # =======数据修正==============
            if cross_order and (not reverse_display):
                if u'总体' not in cross_order:
                    cross_order=cross_order+[u'总体']
                cross_order=[q for q in cross_order if q in plt_data.columns]
                plt_data=pd.DataFrame(plt_data,columns=cross_order)
            if cross_order and reverse_display:
                cross_order=[q for q in cross_order if q in plt_data.index]
                plt_data=pd.DataFrame(plt_data,index=cross_order)
            if 'code_order' in code[qq]:
                code_order=code[qq]['code_order']
                if reverse_display:
                    #code_order=[q for q in code_order if q in t.columns]
                    if u'总体' in t1.columns:
                        code_order=code_order+[u'总体']
                    plt_data=pd.DataFrame(plt_data,columns=code_order)
                else:
                    #code_order=[q for q in code_order if q in t.index]
                    plt_data=pd.DataFrame(plt_data,index=code_order)
            plt_data.fillna(0,inplace=True)
            title='[TOP1]' + title
            if len(plt_data)>max_column_chart:
                prs.add_slide(data={'data':plt_data[::-1],'slide_type':'chart','type':'BAR_CLUSTERED'},\
                              title=title,summary=summary,footnote=footnote)
            else:
                prs.add_slide(data={'data':plt_data,'slide_type':'chart','type':'COLUMN_CLUSTERED'},\
                              title=title,summary=summary,footnote=footnote)




    '''
    # ==============小结页=====================
    difference=pd.Series(difference,index=total_qlist_0)
    '''

    # ========================文件生成和导出======================
    #difference.to_csv('.\\out\\'+filename+u'_显著性检验.csv',encoding='gbk')
    if plt_dstyle:
        filename=filename+'_'+plt_dstyle
    try:
        prs.save('.\\out\\'+filename+u'.pptx')
    except:
        prs.save('.\\out\\'+filename+u'_副本.pptx')
    conclusion.to_excel(Writer,'索引')
    Writer.save()
    if save_dstyle:
        for dstyle in save_dstyle:
            Writer_save[u'Writer_'+dstyle].save()

    return result



def summary_chart(data,code,filename=u'整体统计报告', summary_qlist=None,\
max_column_chart=20,template=None):

    # ===================参数预处理=======================
    if not summary_qlist:
        try:
            summary_qlist=list(sorted(code,key=lambda c: int(re.findall('\d+',c)[0])))
        except:
            summary_qlist=list(code.keys())

    # =================基本数据获取==========================
    #统一的有效样本，各个题目可能有不能的样本数
    sample_len=len(data)

    # ================I/O接口=============================
    # pptx 接口
    prs=rpt.Report(template) if template else rpt.Report()

    if not os.path.exists('.\\out'):
        os.mkdir('.\\out')
    Writer=pd.ExcelWriter('.\\out\\'+filename+'.xlsx')

    result={}#记录每道题的过程数据
    # 记录样本数等信息，用于输出
    conclusion=pd.DataFrame(index=summary_qlist,columns=[u'内容',u'题型',u'样本数'])
    conclusion.to_excel(Writer,u'索引')
    # ================封面页=============================
    prs.add_cover(title=filename)
    # ================背景页=============================
    title=u'说明'
    qtype_count=[code[k]['qtype'] for k in code]
    qtype_count=[[qtype,qtype_count.count(qtype)] for qtype in set(qtype_count)]
    qtype_count=sorted(qtype_count,key=lambda x:x[1],reverse=True)
    summary='该数据一共有{}个题目,其中有'.format(len(code))
    summary+=','.join(['{} {} 道'.format(t[0],t[1]) for t in qtype_count])
    summary+='.\n 经统计, 该数据有效样本数为 {} 份。下表是在该样本数下，各比例对应的置信区间(置信水平95%).'.format(sample_len)
    w=pd.DataFrame(index=[(i+1)*0.05 for i in range(10)],columns=['比例','置信区间'])
    w['比例']=w.index
    w['置信区间']=w['比例'].map(lambda x:confidence_interval(x,sample_len))
    w['置信区间']=w['置信区间'].map(lambda x:'±{:.1f}%'.format(x*100))
    w['比例']=w['比例'].map(lambda x:'{:.0f}% / {:.0f}%'.format(x*100,100-100*x))
    w=w.set_index('比例')
    prs.add_slide(data={'data':w,'slide_type':'table'},title=title,summary=summary)


    for qq in summary_qlist:
        '''
        特殊题型处理
        整体满意度题：后期归为数值类题型
        '''
        #print(qq)
        qtitle=code[qq]['content']
        qlist=code[qq]['qlist']
        qtype=code[qq]['qtype']
        if not(set(qlist) <= set(data.columns)):
            continue
        sample_len_qq=data[code[qq]['qlist']].notnull().T.any().sum()

        conclusion.loc[qq,u'内容']=qtitle
        conclusion.loc[qq,u'题型']=qtype
        conclusion.loc[qq,u'样本数']=sample_len_qq
        # 填空题只统计数据，不绘图
        if qtype == '填空题':
            startcols=0
            for qqlist in qlist:
                tmp=pd.DataFrame(data[qqlist].value_counts()).reset_index()
                tmp.to_excel(Writer,qq,startcol=startcols,index=False)
                startcols+=3
            continue
        if qtype not in [u'单选题',u'多选题',u'排序题',u'矩阵单选题']:
            continue
        try:
            result_t=table(data[qlist],code=code[qq])
        except:
            print(u'脚本处理 {} 时出了一点小问题.....'.format(qq))
            continue
        t=result_t['fop']
        t1=result_t['fo']

        # =======数据修正==============
        if 'code_order' in code[qq]:
            code_order=code[qq]['code_order']
            code_order=[q for q in code_order if q in t.index]
            if u'合计' in t.index:
                code_order=code_order+[u'合计']
            t=pd.DataFrame(t,index=code_order)
            t1=pd.DataFrame(t1,index=code_order)
        t.fillna(0,inplace=True)
        t1.fillna(0,inplace=True)

        # =======保存到Excel中========
        Writer_rows=0
        t2=pd.concat([t,t1],axis=1)
        t2.to_excel(Writer,qq,startrow=Writer_rows,index_label=qq,float_format='%.3f')
        Writer_rows+=len(t2)+2

        # ==========根据个题型提取结论==================
        summary=''
        if qtype in ['单选题','多选题']:
            try:
                gof_result=gof_test(t1)
            except :
                gof_result=-2
            if gof_result==1:
                summary+='拟合优度检验*显著*'
            elif gof_result==0:
                summary+='拟合优度检验*不显著*'
            else:
                summary+='不满足拟合优度检验条件'

        if qtype == '多选题':
            tmp=data[qlist].rename(columns=code[qq]['code'])
            tmp_t=len(tmp)*tmp.shape[1]*np.log(tmp.shape[1])
            if tmp_t<20000:
                minSup=0.08
                minConf=0.40
            elif tmp_t<50000:
                minSup=0.15
                minConf=0.60
            else:
                minSup=0.20
                minConf=0.60
            aso_result,rules,freq=association_rules(tmp,minSup=minSup,minConf=minConf)
            numItem_mean=t1.sum().sum()/sample_len_qq
            if u'合计' in t1.index:
                numItem_mean=numItem_mean/2
            if aso_result:
                summary+=' || 平均每个样本选了{:.1f}个选项 || 找到的关联规则如下(只显示TOP4)：\n{}'.format(numItem_mean,aso_result)
                rules.to_excel(Writer,qq,startrow=Writer_rows,index=False,float_format='%.3f')
                Writer_rows+=len(rules)+2
            else:
                summary+=' || 平均每个样本选了{:.1f}个选项 || 没有找到关联性较大的规则'.format(numItem_mean)

        # 各种题型的结论和相关注释。
        if (qtype in [u'单选题']) and 'fw' in result_t:
            tmp=u'加权平均值'
            if ('name' in code[qq]) and code[qq]['name']==u'满意度':
                    tmp=u'满意度平均值'
            elif ('name' in code[qq]) and code[qq]['name']=='NPS':
                    tmp=u'NPS值'
            summary+=' || {}为：{:.3f}'.format(tmp,result_t['fw'])
        elif qtype =='排序题':
            summary+=' 此处“综合”指标的计算方法为 :={}/总频数.'.format(result_t['weight'])
        if len(summary)==0:
            summary+=u'这里是结论区域'

        # ===============数据再加工==========================
        if qtype in [u'单选题',u'多选题',u'排序题']:
            plt_data=t*100
        else:
            plt_data=t.copy()
        if u'合计' in plt_data.index:
            plt_data.drop([u'合计'],axis=0,inplace=True)
        result[qq]=plt_data
        title=qq+'['+qtype+']: '+qtitle


        footnote=u'数据来源于%s,样本N=%d'%(qq,sample_len_qq)
        # 绘制图表plt_data一般是Series，对于矩阵单选题，其是DataFrame

        if len(t)>max_column_chart:
            prs.add_slide(data={'data':plt_data[::-1],'slide_type':'chart','type':'BAR_CLUSTERED'},\
                          title=title,summary=summary,footnote=footnote)
        elif (len(t)>3) or (len(plt_data.shape)>1 and plt_data.shape[1]>1):
            prs.add_slide(data={'data':plt_data,'slide_type':'chart','type':'COLUMN_CLUSTERED'},\
                          title=title,summary=summary,footnote=footnote)
        else:
            prs.add_slide(data={'data':plt_data,'slide_type':'chart','type':'PIE'},\
                          title=title,summary=summary,footnote=footnote)






        #==============特殊题型处理===============
        # 矩阵单选题特殊处理
        if (qtype == u'矩阵单选题') and ('fw' in result_t):
            plt_data=result_t['fw']
            plt_data.rename(columns={u'加权':u'平均值'},inplace=True)
            plt_data.to_excel(Writer,qq,startrow=Writer_rows,float_format='%.3f')
            Writer_rows=len(plt_data)+2
            plt_data.fillna(0,inplace=True)
            title='[平均值]'+title
            summary=summary+' || 该平均分采用的权值是:\n'+result_t['weight']
            if len(plt_data)>max_column_chart:
                prs.add_slide(data={'data':plt_data[::-1],'slide_type':'chart','type':'BAR_STACKED'},\
                              title=title,summary=summary,footnote=footnote)
            else:
                prs.add_slide(data={'data':plt_data,'slide_type':'chart','type':'COLUMN_STACKED'},\
                              title=title,summary=summary,footnote=footnote)



        # 排序题特殊处理
        if (qtype == u'排序题') and ('TOPN' in result_t):
            plt_data=result_t['TOPN']
            # 将频数和频数百分表保存至本地
            tmp=pd.concat([result_t['TOPN'],result_t['TOPN_fo']],axis=1)
            tmp.to_excel(Writer,qq,startrow=Writer_rows,float_format='%.3f')

            Writer_rows=len(plt_data)+2
            plt_data=plt_data*100
            # =======数据修正==============
            if 'code_order' in code[qq]:
                code_order=code[qq]['code_order']
                #code_order=[q for q in code_order if q in t.index]
                if u'合计' in plt_data.index:
                    code_order=code_order+[u'合计']
                plt_data=pd.DataFrame(plt_data,index=code_order)
            plt_data.fillna(0,inplace=True)

            title='[TOPN]'+title
            if len(plt_data)>max_column_chart:
                prs.add_slide(data={'data':plt_data[::-1],'slide_type':'chart','type':'BAR_STACKED'},\
                              title=title,summary=summary,footnote=footnote)
            else:
                prs.add_slide(data={'data':plt_data,'slide_type':'chart','type':'COLUMN_STACKED'},\
                              title=title,summary=summary,footnote=footnote)



    # ========================文件生成和导出======================
    try:
        prs.save('.\\out\\'+filename+u'.pptx')
    except:
        prs.save('.\\out\\'+filename+u'_副本.pptx')
    conclusion.to_excel(Writer,'索引')
    Writer.save()
    return result

def onekey_gen(data,code,filename=u'reprotgen 报告自动生成',template=None):
    '''一键生成所有可能需要的报告
    包括
    描述统计报告
    单选题的交叉分析报告
    '''
    try:
        summary_chart(data,code,filename=filename,template=template);
    except:
        print('整体报告生成过程中出现错误，将跳过..')
        pass
    print('已生成 '+filename)
    cross_qlist=pre_cross_qlist(data,code)
    if len(cross_qlist)==0:
        return None
    for cross_qq in cross_qlist:
        qq=cross_qq[0]
        cross_order=cross_qq[1]
        if ('name' in code[qq]) and (code[qq]['name']!=''):
            filename='{}_差异分析'.format(code[qq]['name'])
        else:
            filename='{}_差异分析'.format(qq)
        save_dstyle=None #['TGI','CHI']
        try:
            cross_chart(data,code,qq,filename=filename,cross_order=cross_order,\
            save_dstyle=save_dstyle,template=template);
            print('已生成 '+filename)
        except:
            print(filename+'生成过程中出现错误，将跳过...')
            pass
    return None


def scorpion(data,code,filename='scorpion'):
    '''天蝎X计划
    返回一个excel文件
    1、索引
    2、各个题目的频数表
    3、所有可能的交叉分析
    '''

    if not os.path.exists('.\\out'):
        os.mkdir('.\\out')
    Writer=pd.ExcelWriter('.\\out\\'+filename+'.xlsx')
    try:
        qqlist=list(sorted(code,key=lambda c: int(re.findall('\d+',c)[0])))
    except:
        qqlist=list(code.keys())
    qIndex=pd.DataFrame(index=qqlist,columns=[u'content',u'qtype',u'SampleSize'])
    qIndex.to_excel(Writer,u'索引')

    # 生成索引表和频数表
    Writer_rows=0
    for qq in qqlist:
        qtitle=code[qq]['content']
        qlist=code[qq]['qlist']
        qtype=code[qq]['qtype']
        if not(set(qlist) <= set(data.columns)):
            continue
        sample_len_qq=data[code[qq]['qlist']].notnull().T.any().sum()
        qIndex.loc[qq,u'content']=qtitle
        qIndex.loc[qq,u'qtype']=qtype
        qIndex.loc[qq,u'SampleSize']=sample_len_qq
        if qtype not in [u'单选题',u'多选题',u'排序题',u'矩阵单选题']:
            continue
        try:
            result_t=table(data[qlist],code=code[qq])
        except:
            print(u'脚本处理 {} 时出了一点小问题.....'.format(qq))
            continue
        fop=result_t['fop']
        fo=result_t['fo']
        if (qtype == u'排序题') and ('TOPN' in result_t):
            tmp=result_t['TOPN']
            tmp[u'综合']=fo[u'综合']
            fo=tmp.copy()
            tmp=result_t['TOPN_fo']
            tmp[u'综合']=fop[u'综合']
            fop=tmp.copy()
        # =======保存到Excel中========
        fo_fop=pd.concat([fo,fop],axis=1)
        fo_fop.to_excel(Writer,u'频数表',startrow=Writer_rows,startcol=1,index_label=code[qq]['content'],float_format='%.3f')
        tmp=pd.DataFrame({'name':[qq]})
        tmp.to_excel(Writer,u'频数表',index=False,header=False,startrow=Writer_rows)
        Writer_rows+=len(fo_fop)+3
    qIndex.to_excel(Writer,'索引')

    crossAna=pd.DataFrame(columns=['RowVar','ColVar','SampleSize','pvalue','significant','summary'])
    N=0
    qqlist=[qq for qq in qqlist if code[qq]['qtype'] in ['单选题','多选题','矩阵单选题','排序题']]
    start_time=time.clock()
    N_cal=len(qqlist)*(len(qqlist)-1)*0.1# 用于计算脚本剩余时间
    for qq1 in qqlist:
        for qq2 in qqlist:
            #qtype1=code[qq1]['qtype']
            if (N>=N_cal) and (N<N_cal+1.0):
                tmp=(time.clock()-start_time)*9
                if tmp>60:
                    print('请耐心等待, 预计还需要{:.1f}秒'.format(tmp))
            qtype2=code[qq2]['qtype']
            if (qq1==qq2) or (qtype2 not in [u'单选题',u'多选题']):
                continue
            data_index=data[code[qq1]['qlist']]
            data_column=data[code[qq2]['qlist']]
            samplesize=data_column.iloc[list(data_index.notnull().T.any()),:].notnull().T.any().sum()
            try:
                fo=qtable(data,code,qq1,qq2)['fo']
            except :
                crossAna.loc[N,:]=[qq1,qq2,samplesize,'','','']
                N+=1
                continue
            try:
                cdata=contingency(fo,alpha=0.05)
            except :
                crossAna.loc[N,:]=[qq1,qq2,samplesize,'','','']
                N+=1
                continue
            if cdata:
                result=cdata['significant']['result']
                pvalue=cdata['significant']['pvalue']
                summary=cdata['summary']['summary']
            else:
                result=-2
                pvalue=-2
                summary='没有找到结论'
            summary='\n'.join(summary.splitlines()[1:])#去掉第一行
            if len(summary)==0:
                summary='没有找到结论'
            crossAna.loc[N,:]=[qq1,qq2,samplesize,pvalue,result,summary]
            N+=1
    crossAna.to_excel(Writer,'交叉分析表',index=False)

    Writer.save()
