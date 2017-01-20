# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 20:05:36 2016
@author: gason
"""
'''
pptx 用的单位是pptx.util.Emu,  英语单位
pptx.util.Inches(1)=914400
pptx.util.Pt(1)=12700
pptx.util.Cm(1)=360000
slide的大小:
pptx.Presentation().slide_height
pptx.Presentation().slide_width
# 对于python2，源代码有一些的bug
#pptx\\chart\\xmlwriter.py
1338和1373的：escape(str(name))改为escape(unicode(name))
'''

import os
import re
import sys



import pandas as pd
import numpy as np

from pptx import Presentation
from pptx.chart.data import ChartData,XyChartData,BubbleChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches, Pt, Emu
from pptx.enum.chart import XL_LEGEND_POSITION
from pptx.enum.chart import XL_LABEL_POSITION


def df_to_table(slide,df,left,top,width,height,rownames=False,colnames=True):
    df=pd.DataFrame(df)
    rows, cols = df.shape
    res = slide.shapes.add_table(rows+colnames, cols+rownames, left, top, width, height)
    '''
    for c in range(cols+rownames):
        res.table.columns[c].width = colwidth
    '''
    # Insert the column names
    if colnames:
        for col_index, col_name in enumerate(list(df.columns)):
            res.table.cell(0,col_index+rownames).text = '%s'%(col_name)
    if rownames:
        for col_index, col_name in enumerate(list(df.index)):
            res.table.cell(col_index+colnames,0).text = '%s'%(col_name)
    m = df.as_matrix()
    for row in range(rows):
        for col in range(cols):
            res.table.cell(row+colnames, col+rownames).text = '%s'%(m[row, col])

def plot_table(prs,df,layouts=[0,5],title=u'我是标题',summary=u'我是简短的结论'):
    slide_width=prs.slide_width
    slide_height=prs.slide_height
    # 可能需要修改以适应更多的情形
    title_only_slide = prs.slide_masters[layouts[0]].slide_layouts[layouts[1]]
    slide = prs.slides.add_slide(title_only_slide)
    #title=u'这里是标题'
    slide.shapes.title.text = title
    left,top = Emu(0.05*slide_width), Emu(0.10*slide_height)
    width,height = Emu(0.7*slide_width), Emu(0.1*slide_height)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    #summary=u'这里是一些简短的结论'
    txBox.text_frame.text=summary
    # 绘制表格
    left=Emu(0.10*slide_width)
    top=Emu(0.20*slide_height)
    width=Emu(0.8*slide_width)
    height=Emu(0.7*slide_height)
    df_to_table(slide,df,left,top,width,height,rownames=True)



def df_to_chartdata(df,datatype,number_format=None):
    '''
    根据给定的图表数据类型生成相应的数据
    Chartdata:一般的数据
    XyChartData: 散点图数据
    BubbleChartData:气泡图数据
    '''
    df=pd.DataFrame(df)
    datatype=datatype.lower()
    if datatype == 'chartdata':
        chart_data = ChartData()
        chart_data.categories = ['%s'%(c) for c in list(df.index)]
        for col_name in df.columns:
            chart_data.add_series('%s'%(col_name),list(df[col_name]),number_format)
        return chart_data
    if datatype == 'xychartdata':
        chart_data=XyChartData()
        if not isinstance(df,list):
            df=[df]
        for d in df:
            series_name='%s'%(d.columns[0])+' vs '+'%s'%(d.columns[1])
            series_ = chart_data.add_series(series_name)
            for i in range(len(d)):
                series_.add_data_point(d.iloc[i,0], d.iloc[i,1])
        return chart_data
    if datatype == 'bubblechartdata':
        chart_data=BubbleChartData()
        if not isinstance(df,list):
            df=[df]
        for d in df:
            series_name='%s'%(d.columns[0])+' vs '+'%s'%(d.columns[1])
            series_ = chart_data.add_series(series_name)
            for i in range(len(d)):
                series_.add_data_point(d.iloc[i,0],d.iloc[i,1],d.iloc[i,2])
        return chart_data

def plot_chart(prs,df,chart_type,title=u'我是标题',summary=u'我是简短的结论',\
footnote=None,chart_format=None,layouts=[0,5]):
    '''
    直接将数据绘制到一张ppt上，且高度定制化
    默认都有图例，且图例在下方
    默认都有数据标签
    '''

    chart_list={\
    "AREA":[1,"ChartData"],\
    "AREA_STACKED":[76,"ChartData"],\
    "AREA_STACKED_100":[77,"ChartData"],\
    "THREE_D_AREA":[-4098,"ChartData"],\
    "THREE_D_AREA_STACKED":[78,"ChartData"],\
    "THREE_D_AREA_STACKED_100":[79,"ChartData"],\
    "BAR_CLUSTERED":[57,"ChartData"],\
    "BAR_TWO_WAY":[57,"ChartData"],\
    "BAR_OF_PIE":[71,"ChartData"],\
    "BAR_STACKED":[58,"ChartData"],\
    "BAR_STACKED_100":[59,"ChartData"],\
    "THREE_D_BAR_CLUSTERED":[60,"ChartData"],\
    "THREE_D_BAR_STACKED":[61,"ChartData"],\
    "THREE_D_BAR_STACKED_100":[62,"ChartData"],\
    "BUBBLE":[15,"BubbleChartData"],\
    "BUBBLE_THREE_D_EFFECT":[87,"BubbleChartData"],\
    "COLUMN_CLUSTERED":[51,"ChartData"],\
    "COLUMN_STACKED":[52,"ChartData"],\
    "COLUMN_STACKED_100":[53,"ChartData"],\
    "THREE_D_COLUMN":[-4100,"ChartData"],\
    "THREE_D_COLUMN_CLUSTERED":[54,"ChartData"],\
    "THREE_D_COLUMN_STACKED":[55,"ChartData"],\
    "THREE_D_COLUMN_STACKED_100":[56,"ChartData"],\
    "CYLINDER_BAR_CLUSTERED":[95,"ChartData"],\
    "CYLINDER_BAR_STACKED":[96,"ChartData"],\
    "CYLINDER_BAR_STACKED_100":[97,"ChartData"],\
    "CYLINDER_COL":[98,"ChartData"],\
    "CYLINDER_COL_CLUSTERED":[92,"ChartData"],\
    "CYLINDER_COL_STACKED":[93,"ChartData"],\
    "CYLINDER_COL_STACKED_100":[94,"ChartData"],\
    "DOUGHNUT":[-4120,"ChartData"],\
    "DOUGHNUT_EXPLODED":[80,"ChartData"],\
    "LINE":[4,"ChartData"],\
    "LINE_MARKERS":[65,"ChartData"],\
    "LINE_MARKERS_STACKED":[66,"ChartData"],\
    "LINE_MARKERS_STACKED_100":[67,"ChartData"],\
    "LINE_STACKED":[63,"ChartData"],\
    "LINE_STACKED_100":[64,"ChartData"],\
    "THREE_D_LINE":[-4101,"ChartData"],\
    "PIE":[5,"ChartData"],\
    "PIE_EXPLODED":[69,"ChartData"],\
    "PIE_OF_PIE":[68,"ChartData"],\
    "THREE_D_PIE":[-4102,"ChartData"],\
    "THREE_D_PIE_EXPLODED":[70,"ChartData"],\
    "PYRAMID_BAR_CLUSTERED":[109,"ChartData"],\
    "PYRAMID_BAR_STACKED":[110,"ChartData"],\
    "PYRAMID_BAR_STACKED_100":[111,"ChartData"],\
    "PYRAMID_COL":[112,"ChartData"],\
    "PYRAMID_COL_CLUSTERED":[106,"ChartData"],\
    "PYRAMID_COL_STACKED":[107,"ChartData"],\
    "PYRAMID_COL_STACKED_100":[108,"ChartData"],\
    "RADAR":[-4151,"ChartData"],\
    "RADAR_FILLED":[82,"ChartData"],\
    "RADAR_MARKERS":[81,"ChartData"],\
    "STOCK_HLC":[88,"ChartData"],\
    "STOCK_OHLC":[89,"ChartData"],\
    "STOCK_VHLC":[90,"ChartData"],\
    "STOCK_VOHLC":[91,"ChartData"],\
    "SURFACE":[83,"ChartData"],\
    "SURFACE_TOP_VIEW":[85,"ChartData"],\
    "SURFACE_TOP_VIEW_WIREFRAME":[86,"ChartData"],\
    "SURFACE_WIREFRAME":[84,"ChartData"],\
    "XY_SCATTER":[-4169,"XyChartData"],\
    "XY_SCATTER_LINES":[74,"XyChartData"],\
    "XY_SCATTER_LINES_NO_MARKERS":[75,"XyChartData"],\
    "XY_SCATTER_SMOOTH":[72,"XyChartData"],\
    "XY_SCATTER_SMOOTH_NO_MARKERS":[73,"XyChartData"]}

    slide_width=prs.slide_width
    slide_height=prs.slide_height
    # 可能需要修改以适应更多的情形
    # layouts[0]代表第几个母版，layouts[1]代表母版中的第几个版式
    title_only_slide = prs.slide_masters[layouts[0]].slide_layouts[layouts[1]]
    slide = prs.slides.add_slide(title_only_slide)
    #title=u'这里是标题'
    slide.shapes.title.text = title
    left,top = Emu(0.05*slide_width), Emu(0.10*slide_height)
    width,height = Emu(0.7*slide_width), Emu(0.1*slide_height)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    #summary=u'这里是一些简短的结论'
    txBox.text_frame.text=summary
    chart_data=df_to_chartdata(df,chart_list[chart_type][1])
    x, y = Emu(0.05*slide_width), Emu(0.20*slide_height)
    cx, cy = Emu(0.85*slide_width), Emu(0.70*slide_height)
    chart=slide.shapes.add_chart(chart_list[chart_type.upper()][0], \
    x, y, cx, cy, chart_data).chart
    font_default_size=Pt(10)
    # 添加图例
    chart.has_legend = True
    chart.legend.font.size=font_default_size
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False

    try:
        chart.category_axis.tick_labels.font.size=font_default_size
    except Exception as e:
        unsuc=0#暂时不知道怎么处理
    try:
        chart.value_axis.tick_labels.font.size=font_default_size
    except Exception as e:
        unsuc=0
    # 添加数据标签
    '''
    non_available_list=['BUBBLE','BUBBLE_THREE_D_EFFECT','XY_SCATTER',\
    'XY_SCATTER_LINES']
    if chart_type not in non_available_list:
        plot = chart.plots[0]
        plot.has_data_labels = True
        plot.data_labels.font.size = font_default_size
        #data_labels = plot.data_labels
        #data_labels.position = XL_LABEL_POSITION.BEST_FIT
    '''

    # 自定义format
    if chart_format:
        for k in chart_format:
            exec('chart.'+k+'='+'%s'%(chart_format[k]))

    '''
    if chart_type == 'BAR_TWO_WAY':
        chart
    '''

def plot_textbox(prs,layouts=[0,5],title=u'我是文本框页标题',summary=u'我是内容'):
    '''
    只绘制一个文本框，用于目录、小结等
    '''
    slide_width=prs.slide_width
    slide_height=prs.slide_height
    # 可能需要修改以适应更多的情形
    title_only_slide = prs.slide_masters[layouts[0]].slide_layouts[layouts[1]]
    slide = prs.slides.add_slide(title_only_slide)
    #title=u'这里是标题'
    slide.shapes.title.text = title
    left,top = Emu(0.15*slide_width), Emu(0.10*slide_height)
    width,height = Emu(0.7*slide_width), Emu(0.7*slide_height)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    txBox.text_frame.text=summary

def read_code(filename):
    d=pd.read_excel(filename,header=None)
    d.replace({np.nan:'NULL'},inplace=True)
    d=d.as_matrix()
    code={}
    for i in range(len(d)):
        tmp=d[i,0]
        if tmp == 'key':
            code[d[i,1]]={}
            key=d[i,1]
        elif tmp in ['qlist','code_order']:
            ind=np.argwhere(d[i+1:,0]!='NULL')
            if ind.any():
                j=i+1+ind[0][0]
            else:
                j=len(d)-1
            qlist=list(d[i:j,1])
            code[key][tmp]=qlist
        elif tmp in ['code','code_r']:
            ind=np.argwhere(d[i+1:,0]!='NULL')
            if ind.any():
                j=i+1+ind[0][0]
            else:
                j=len(d)-1
            tmp1=list(d[i:j,1])
            tmp2=list(d[i:j,2])
            code[key][tmp]=dict(zip(tmp1,tmp2))
        elif tmp == 'NULL':
            continue
        else:
            code[key][tmp]=d[i,1]
    return code

def to_code(code,savename='code.xlsx',method='xlsx'):
    method=method.lower()
    if method != 'xlsx':
        return
    tmp=pd.DataFrame(columns=['name','value1','value2'])
    i=0
    if all(['Q' in c[0] for c in code.keys()]):
        key_qlist=sorted(code,key=lambda c:int(c[1:]))
    else:
        key_qlist=code.keys()
    for key in key_qlist:
        code0=code[key]
        tmp.loc[i]=['key',key,'']
        i+=1
        for key0 in code0:
            tmp2=code0[key0]
            tmp.loc
            if type(tmp2) == list:
                tmp.loc[i]=[key0,tmp2[0],'']
                i+=1
                for ll in tmp2[1:]:
                    tmp.loc[i]=['',ll,'']
                    i+=1
            elif type(tmp2) == dict:
                j=0
                for key1 in tmp2.keys():
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
        tmp.to_excel(savename,index=False,header=False)
    else:
        tmp.to_csv(savename,index=False,header=False)


def wenjuanwang(filepath='.\\data'):
    if isinstance(filepath,list):
        filename1=filepath[0]
        filename2=filepath[1]
        filename2=filepath[2]
    elif os.path.isdir(filepath):
        filename1=os.path.join(filepath,'All_Data_Readable.csv')
        filename2=os.path.join(filepath,'All_Data_Original.csv')
        filename3=os.path.join(filepath,'code.csv')
    else:
        print('can not dection the filepath!')

    d1=pd.read_csv(filename1,encoding='gbk')
    d1.drop([u'答题时长'],axis=1,inplace=True)
    d2=pd.read_csv(filename2,encoding='gbk')
    d3=pd.read_csv(filename3,encoding='gbk',header=None,na_filter=False)
    d3=d3.as_matrix()

    '''
    对每一个题目的情形进行编码：题目默认按照Q1、Q2等给出
    Qn.content: 题目内容
    Qn.qtype: 题目类型，包含:单选题、多选题、填空题、排序题、矩阵单选题等
    Qn.qlist: 题目列表，例如多选题对应着很多小题目
    Qn.code: 题目选项编码
    Qn.code_r: 下题目对应的编码(矩阵题目专有)
    Qn.qtype2: 特殊类型，包含：城市题、NPS题等
    '''
    code={}
    for i in range(len(d3)):
        if d3[i,0]:
            key=d3[i,0]
            code[key]={}
            code[key]['content']=d3[i,1]
            code[key]['qtype']=d3[i,2]
            code[key]['code']={}
        elif d3[i,2]:
            tmp=d3[i,1]
            if code[key]['qtype']  in [u'多选题',u'排序题']:
                tmp=key+'_A'+'%s'%(tmp)
                code[key]['code'][tmp]='%s'%(d3[i,2])
            else:
                try:
                    tmp=np.float(tmp)
                except:
                    tmp='%s'%(tmp)
                code[key]['code'][tmp]='%s'%(d3[i,2])

    qnames_Readable=list(d1.columns)
    qnames=list(d2.columns)
    for key in code.keys():
        qlist=[]
        for name in qnames:
            if re.match(key+'_',name) or key==name:
                qlist.append(name)
        code[key]['qlist']=qlist
        code[key]['code_r']={}
        if code[key]['qtype']  in [u'矩阵单选题']:
            tmp=[qnames_Readable[qnames.index(q)] for q in code[key]['qlist']]
            code_r=[re.findall('_([^_]*?)$',t)[0] for t in tmp]
            code[key]['code_r']=dict(zip(code[key]['qlist'],code_r))
    return (d2,code)


def wenjuanxing(filepath='.\\data',headlen=6):
    #headlen=6# 问卷从开始到第一道正式题的数目（一般包含序号，提交答卷时间的等等）
    if isinstance(filepath,list):
        filename1=filepath[0]
        filename2=filepath[1]
    elif os.path.isdir(filepath):
        filename1=os.path.join(filepath,'All_Data_Readable.xls')
        filename2=os.path.join(filepath,'All_Data_Original.xls')
    else:
        print('can not dection the filepath!')

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
                code[new_name]['qtype']=u'多选题'
            elif '→' in qcontent:
                code[new_name]['qtype']=u'排序题'
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
            code[current_name]['sample_len']=d2[current_name].notnull().sum()
            #code[current_name]['qtype']=u'单选题'
            c1=d1[current_name].unique()
            c2=d2[current_name].unique()
            '''
            if (c2.dtype != object) and len(c2)<code[current_name]['sample_len']*0.6:
                code[current_name]['code']=dict(zip(c2,c1))
                code[current_name]['qtype']=u'单选题'
            else:
                code[current_name]['qtype']=u'填空题'
            '''    
            if (c2.dtype == object) or (list(c1)==list(c2)):
                code[current_name]['qtype']=u'填空题'
            else:
                code[current_name]['qtype']=u'单选题'
                code[current_name]['code']=dict(zip(c2,c1)) 
        elif tmp2:
            name0='Q'+tmp2[0]
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
                code[current_name]['sample_len']=d2[name].notnull().sum()
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
            tmp3=re.findall(u'第.*?题\((.*?)\)',name)[0]
            if code[current_name]['qtype'] == u'矩阵单选题':
                code[current_name]['code_r'][name1]=tmp3
            else:
                code[current_name]['code'][name1]=tmp3
        # 删除字典中的nan
        keys=list(code[current_name]['code'].keys())
        for key in keys:
            if '%s'%key == 'nan':
                del  code[current_name]['code'][key]
    return (d2,code)



if __name__ == '__main__':

    prs = Presentation()
    slide_width=prs.slide_width
    slide_height=prs.slide_height

    '''
    模板文件放在,可自己修改，其中不能增加板式
    C:\Anaconda2\Lib\site-packages\pptx\templates\default.pptx
    复杂用法：
    prs = Presentation()
    '''

    #标题页面
    title_slide = prs.slide_layouts[0]
    #重点列表
    bullet_slide = prs.slide_layouts[1]
    # 只有一个标题
    title_only_slide = prs.slide_layouts[5]
    # 空页面
    blank_slide = prs.slide_layouts[6]

    # 第一页：标题
    slide = prs.slides.add_slide(title_slide)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = u"半自动化报告生成"
    subtitle.text = "powered by python-pptx"

    # 第二页：文本框
    slide = prs.slides.add_slide(blank_slide)
    left = top = width = height = Inches(1)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.text = "This is text inside a textbox"
    p = tf.add_paragraph()
    p.text = "This is a second paragraph that's bold"
    p.font.bold = True
    p = tf.add_paragraph()
    p.text = "This is a third paragraph that's big"
    p.font.size = Pt(40)

    # 第三页：表格
    '''
    默认表格样式
    '''
    slide = prs.slides.add_slide(title_only_slide)
    shapes = slide.shapes
    shapes.title.text = u'表格示例'
    #d=pd.read_csv('data.csv',encoding='gbk')
    #t=pd.crosstab(d['Q2'],d['Q3'])
    left=Emu(0.15*slide_width)
    top=Emu(0.3*slide_height)
    width=Emu(0.7*slide_width)
    height=Emu(0.6*slide_height)
    #df_to_table(slide,t,left,top,width,height,rownames=True)

    # 第4、5、6、7页：图表

    t1=pd.DataFrame({u'满意度':[2,4,3,3.5],u'重要度':[2.2,3.5,2.6,3.8]},columns=[u'满意度',u'重要度'])
    t2=pd.DataFrame({u'满意度':[2,4,3,3.5],u'重要度':[3.2,4.5,3.6,4.8]},columns=[u'满意度',u'重要度'])
    t3=pd.DataFrame({u'满意度':[2,4,3,3.5],u'重要度':[2.2,3.5,2.6,3.8],\
    u'大小':[3,2,6,8]},columns=[u'满意度',u'重要度',u'大小'])
    t1.rename(index={0:u'张',1:u'王',2:u'李',3:u'宋'},inplace=True)
    plot_chart(prs,t1,'COLUMN_CLUSTERED')
    plot_chart(prs,t1,'COLUMN_STACKED')
    plot_chart(prs,t1,'BAR_CLUSTERED')
    #plot_chart(prs,t,'BAR_CLUSTERED')
    plot_chart(prs,[t1,t2],'XY_SCATTER')
    plot_chart(prs,t3,'BUBBLE')
    plot_chart(prs,pd.DataFrame({u'满意度':[2,4,3,3.5]}),'PIE')

    # 第8页：自定义图
    slide = prs.slides.add_slide(title_only_slide)
    slide.shapes.title.text = u'高度定制化'
    left,top = Emu(0.12*slide_width), Emu(0.2*slide_height)
    width,height = Emu(0.7*slide_width), Emu(0.1*slide_height)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    txBox.text_frame.text=u'这里是一些简短的结论'
    # define chart data ---------------------
    '''
    chart_data = ChartData()
    chart_data.categories = ['East', 'West', 'Midwest']
    chart_data.add_series('Series 1', (19.2, 21.4, 16.7))
    chart_data.add_series('Series 2', (23, 16, 19))
    '''
    chart_data=df_to_chartdata(t1,'ChartData')
    # add chart to slide --------------------
    x, y = Emu(0.05*slide_width), Emu(0.35*slide_height)
    cx, cy = Emu(0.76*slide_width), Emu(0.55*slide_height)
    chart=slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, \
    x, y, cx, cy, chart_data).chart



    '''
    # ----样式-------
    chart.chart_style: 样式(ppt自带的1到48)
    # ----坐标轴-------
    axis=chart.category_axis: X轴坐标控制
    axis=chart.value_axis   :Y周坐标控制
    axis.visible: 坐标轴是否可见
    axis.has_major_gridlines：添加主要网格线
    axis.major_gridlines: 设置主要网格线
    axis.has_minor_gridlines: 添加次要网格线
    axis.minor_gridlines: 设置次要网格线
    axis.major_tick_mark: 主要刻度线类型(XL_TICK_MARK.OUTSIDE: 无，内部，外部，交叉)
    axis.minor_tick_mark: 主要刻度线类型(XL_TICK_MARK.OUTSIDE: 无，内部，外部，交叉)
    axis.maximum_scale:  最大值
    axis.minimum_scale：最小值
    axis.tick_label_position: 坐标轴标签
    axis.tick_labels.font:  字体设置(共8个维度,如.bold=True, .size=Pt(12))
    axis.tick_labels.number_format: 数字格式('0"%"')
    axis.tick_labels.number_format_is_linked
    axis.tick_labels.offset
    # ----数据标签-------
    plot = chart.plots[0]
    plot.has_data_labels = True
    data_labels = plot.data_labels
    data_labels.font.size = Pt(13)
    # from pptx.dml.color import RGBColor
    data_labels.font.color.rgb = RGBColor(0x0A, 0x42, 0x80)
    # from pptx.enum.chart import XL_LABEL_POSITION
    data_labels.position = XL_LABEL_POSITION.INSIDE_END
    # ----图例-------
    # from pptx.enum.chart import XL_LEGEND_POSITION
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.RIGHT
    chart.legend.include_in_layout = False
    '''
    '''
    category_axis = chart.category_axis
    category_axis.has_major_gridlines = True
    category_axis.minor_tick_mark = XL_TICK_MARK.OUTSIDE
    category_axis.tick_labels.font.italic = True
    category_axis.tick_labels.font.size = Pt(24)
    value_axis = chart.value_axis
    value_axis.maximum_scale = 50.0
    value_axis.minor_tick_mark = XL_TICK_MARK.OUTSIDE
    value_axis.has_minor_gridlines = True
    tick_labels = value_axis.tick_labels
    tick_labels.number_format = '0"%"'
    tick_labels.font.bold = True
    tick_labels.font.size = Pt(14)
    '''

    # 添加致谢页
    #slide = prs.slides.add_slide(prs.slide_layouts[8])
    prs.save('test.pptx')
