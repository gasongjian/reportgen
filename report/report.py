# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 20:05:36 2016
@author: JSong
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
import math



import pandas as pd
import numpy as np

from pptx import Presentation
from pptx.chart.data import ChartData,XyChartData,BubbleChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches, Pt, Emu
from pptx.enum.chart import XL_LEGEND_POSITION
from pptx.enum.chart import XL_LABEL_POSITION
from pptx.dml.color import RGBColor





def df_to_table(slide,df,left,top,width,height,index_names=False,columns_names=True):
    '''将pandas数据框添加到slide上，并生成pptx上的表格
    输入：
    slide：PPT的一个页面，由pptx.Presentation().slides.add_slide()给定
    df：需要转换的数据框
    lef,top: 表格在slide中的位置
    width,height: 表格在slide中的大小
    index_names: Bool,是否需要显示行类别的名称
    columns_names: Bool,是否需要显示列类别的名称
    返回：
    返回带表格的slide
    '''
    df=pd.DataFrame(df)
    rows, cols = df.shape
    res = slide.shapes.add_table(rows+columns_names, cols+index_names, left, top, width, height)
    '''
    for c in range(cols+rownames):
        res.table.columns[c].width = colwidth
    '''
    # Insert the column names
    if columns_names:
        for col_index, col_name in enumerate(list(df.columns)):
            res.table.cell(0,col_index+index_names).text = '%s'%(col_name)
    if index_names:
        for col_index, col_name in enumerate(list(df.index)):
            res.table.cell(col_index+columns_names,0).text = '%s'%(col_name)
    m = df.as_matrix()
    for row in range(rows):
        for col in range(cols):
            res.table.cell(row+columns_names, col+index_names).text = '%s'%(m[row, col])

def plot_table(prs,df,layouts=[0,5],title=u'我是标题',summary=u'我是简短的结论'):
    '''根据给定的数据，在给定的prs上新增一页表格ppt
    输入：
    prs: PPT文件接口
    df: 数据框
    layouts: [0]为PPT母版顺序，[1]为母版内的版式顺序
    输出：
    更新后的prs
    '''
    df=pd.DataFrame(df)
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
    '''添加自适应的表格大小
    默认最大12*6，width=0.80,height=0.70
    left=0.1,top=0.25
    '''
    R,C=df.shape
    width=max(0.5,min(1,C/6.0))*0.80
    height=max(0.5,min(1,R/12.0))*0.70
    left=0.5-width/2
    top=(1-height)*2/3
    left=Emu(left*slide_width)
    top=Emu(top*slide_height)
    width=Emu(width*slide_width)
    height=Emu(height*slide_height)
    df_to_table(slide,df,left,top,width,height,index_names=True)



def df_to_chartdata(df,datatype,number_format=None):
    '''
    根据给定的图表数据类型生成相应的数据
    Chartdata:一般的数据
    XyChartData: 散点图数据
    BubbleChartData:气泡图数据
    '''
    if isinstance(df,pd.Series):
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
footnote=None,chart_format=None,layouts=[0,5],has_data_labels=True):
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
    # 添加标题 title=u'这里是标题'
    slide.shapes.title.text = title
    # 添加结论 summary=u'这里是一些简短的结论'
    left,top = Emu(0.05*slide_width), Emu(0.10*slide_height)
    width,height = Emu(0.7*slide_width), Emu(0.1*slide_height)
    txBox = slide.shapes.add_textbox(left, top, width, height)
    txBox.text_frame.text=summary
    try:
        txBox.text_frame.fit_text(max_size=12)
    except:
        log='cannot fit the size of font'

    # 添加脚注 footnote=u'这里是脚注'
    if footnote:
        left,top = Emu(0.02*slide_width), Emu(0.95*slide_height)
        width,height = Emu(0.70*slide_width), Emu(0.05*slide_height)
        txBox = slide.shapes.add_textbox(left, top, width, height)
        txBox.text_frame.text=footnote
        try:
            txBox.text_frame.fit_text(max_size=10)
        except:
            log='cannot fit the size of font'
    chart_type_code=chart_list[chart_type][1]
    chart_data=df_to_chartdata(df,chart_type_code)
    x, y = Emu(0.05*slide_width), Emu(0.20*slide_height)
    cx, cy = Emu(0.85*slide_width), Emu(0.70*slide_height)
    chart=slide.shapes.add_chart(chart_list[chart_type.upper()][0], \
    x, y, cx, cy, chart_data).chart

    if chart_type_code in [-4169,72,73,74,75]:
        return

    font_default_size=Pt(10)
    # 添加图例
    if (df.shape[1]>1) or (chart_type=='PIE'):
        chart.has_legend = True
        chart.legend.font.size=font_default_size
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False

    try:
        chart.category_axis.tick_labels.font.size=font_default_size
    except:
        unsuc=0#暂时不知道怎么处理
    try:
        chart.value_axis.tick_labels.font.size=font_default_size
    except:
        unsuc=0
    # 添加数据标签

    non_available_list=['BUBBLE','BUBBLE_THREE_D_EFFECT','XY_SCATTER',\
    'XY_SCATTER_LINES','PIE']
    # 大致检测是否采用百分比
    if (df.sum()>=80).any() and (df<=100).any().any():
        # 数据条的数据标签格式
        number_format1='0.0"%"'
        # 坐标轴的数据标签格式
        number_format2='0"%"'
    else:
        number_format1='0.00'
        number_format2='0.0'

    if (chart_type not in non_available_list) or (chart_type == 'PIE'):
        plot = chart.plots[0]
        plot.has_data_labels = True
        plot.data_labels.font.size = font_default_size
        plot.data_labels.number_format = number_format1
        #plot.data_labels.number_format_is_linked=True
        #data_labels = plot.data_labels
        #plot.data_labels.position = XL_LABEL_POSITION.BEST_FIT
    if (chart_type not in non_available_list):
        #chart.value_axis.maximum_scale = 1
        if df.shape[1]==1:
            chart.value_axis.has_major_gridlines = False
        else:
            chart.value_axis.has_major_gridlines = True
        tick_labels = chart.value_axis.tick_labels
        tick_labels.number_format = number_format2
        tick_labels.font.size = font_default_size

    # 修改纵坐标格式
    '''
    tick_labels = chart.value_axis.tick_labels
    tick_labels.number_format = '0"%"'
    tick_labels.font.bold = True
    tick_labels.font.size = Pt(10)
    '''

    # 填充系列的颜色
    ''' 最好的方法还是修改母版文件中的主题颜色，这里只提供方法
    if df.shape[1]==1:
        chart.series[0].fill()
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
    d.replace({np.nan:'NULL'},inplace=True)
    d=d.as_matrix()
    code={}
    for i in range(len(d)):
        tmp=d[i,0]
        if tmp == 'key':
            # 识别题号
            code[d[i,1]]={}
            key=d[i,1]
        elif tmp in ['qlist','code_order']:
            # 识别字典值为列表的字段
            ind=np.argwhere(d[i+1:,0]!='NULL')
            if ind.any():
                j=i+1+ind[0][0]
            else:
                j=len(d)-1
            code[key][tmp]=list(d[i:j,1])
        elif tmp in ['code','code_r']:
            # 识别字典值为字典的字段
            ind=np.argwhere(d[i+1:,0]!='NULL')
            if ind.any():
                j=i+1+ind[0][0]
            else:
                j=len(d)
            tmp1=list(d[i:j,1])
            tmp2=list(d[i:j,2])
            code[key][tmp]=dict(zip(tmp1,tmp2))
        # 识别其他的列表字段
        elif (tmp!='NULL') and (d[i,2]=='NULL') and ((i==len(d)-1) or (d[i+1,0]=='NULL')):
            ind=np.argwhere(d[i+1:,0]!='NULL')
            if ind.any():
                j=i+1+ind[0][0]
            else:
                j=len(d)
            if i==len(d)-1:
                code[key][tmp]=d[i,1]
            else:
                code[key][tmp]=list(d[i:j,1])            
        # 识别其他的字典字段
        elif (tmp!='NULL') and (d[i,2]!='NULL') and ((i==len(d)-1) or (d[i+1,0]=='NULL')):
            ind=np.argwhere(d[i+1:,0]!='NULL')
            if ind.any():
                j=i+1+ind[0][0]
            else:
                j=len(d)
            tmp1=list(d[i:j,1])
            tmp2=list(d[i:j,2])
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
        for key0 in code0:
            tmp2=code0[key0]
            if type(tmp2) == list:
                tmp.loc[i]=[key0,tmp2[0],'']
                i+=1
                for ll in tmp2[1:]:
                    tmp.loc[i]=['',ll,'']
                    i+=1
            elif type(tmp2) == dict:
                try:
                    tmp2_key=sorted(tmp2,key=lambda c:int(re.findall('\d+','%s'%c)[-1]))
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
Qn.code: 题目选项编码
Qn.code_r: 题目对应的编码(矩阵题目专有)
Qn.code_order: 题目类别的顺序，用于PPT报告的生成[一般后期添加]
Qn.name: 特殊类型，包含：城市题、NPS题等
'''

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
        列表，[0]为按文本数据路径，[1]为按序号文本
        文件夹路径，函数会自动在文件夹下搜寻相关数据，优先为\d+_\d+_0.xls和\d+_\d+_2.xls
    headlen: 问卷星数据基础信息的列数
    输出：
    (data,code):
        data为按序号的数据，题目都替换成了Q_n
        code为数据编码，可利用函数to_code()导出为json格式或者Excel格式数据
    '''
    #headlen=6# 问卷从开始到第一道正式题的数目（一般包含序号，提交答卷时间的等等）
    if isinstance(filepath,list):
        filename1=filepath[0]
        filename2=filepath[1]
    elif os.path.isdir(filepath):
        filelist=os.listdir(filepath)
        for f in filelist:
            s1=re.findall('\d+_\d+_0.xls',f)
            s2=re.findall('\d+_\d+_2.xls',f)
            if s1:
                filename1=s1[0]
            if s2:
                filename2=s2[0]
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
            if (c2.dtype == object) or (list(c1)==list(c2)) or (len(c2)>50):
                code[current_name]['qtype']=u'填空题'
            else:
                code[current_name]['qtype']=u'单选题'
                code[current_name]['code']=dict(zip(c2,c1))
                if 'qlist_open' in code[current_name].keys():
                    tmp=d1[current_name].map(lambda x: re.findall('〖(.*?)〗',x)[0] if re.findall('〖(.*?)〗',x) else '')
                    ind=np.argwhere(d2.columns.values==current_name).tolist()[0][0]
                    d2.insert(ind+1,current_name+'_open',tmp)
                    c1=d1[current_name].map(lambda x: re.sub('〖.*?〗','',x)).unique()
                    code[current_name]['qlist_open']=[current_name+'_open']
                code[current_name]['code']=dict(zip(c2,c1))

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
    return (d2,code)

## ===========================================================
#
#
#                     数据分析和输出                          #
#
#
## ==========================================================



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
            if qtype == u'单选题':
                data1[qq].replace(code[qq]['code'],inplace=True)
            elif qtype == u'矩阵单选题':
                data1[code[qq]['qlist']].replace(code[qq]['code'],inplace=True)
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
    '''
    if isinstance(data,pd.core.frame.DataFrame):
        data=data[data.columns[0]]
    categorys=sorted(data[data.notnull()].unique())
    categorys=data[data.notnull()].unique()
    try:
        categorys=sorted(categorys)
    except:
        print('sa_to_ma function::cannot sorted')
    data_ma=pd.DataFrame(index=data.index,columns=categorys)
    for c in categorys:
        data_ma[c]=data.map(lambda x : int(x==c))
    data_ma.loc[data.isnull(),:]=np.nan
    return data_ma


def binomial_interval(p,n,alpha=0.05):
    import scipy.stats as stats
    a=p-stats.norm.ppf(1-alpha/2)*math.sqrt(p*(1-p)/n)
    b=p+stats.norm.ppf(1-alpha/2)*math.sqrt(p*(1-p)/n)
    return (a,b)

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
    
    
def mca(X,N=2):
    '''对应分析函数，暂时支持双因素
    X：观察频数表
    N：返回的维数，默认2维
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

def sankey(df,filename=None):
    '''SanKey图绘制
    注:暂时没找到好的Python方法，所以只生成R语言所需数据
    返回links 和 nodes
    # R code 参考
    library(networkD3)
    dd=read.csv('price_links.csv')
    links<-data.frame(source=dd$from,target=dd$to,value=dd$value)
    nodes=read.csv('price_nodes.csv',header = FALSE)
    names(nodes)='name'
    Energy=c(links=links,nodes=nodes)
    sankeyNetwork(Links = links, Nodes = nodes, Source = "source",
                  Target = "target", Value = "value", NodeID = "name",
                  units = "TWh",fontSize = 18,fontFamily='微软雅黑',nodeWidth=20) 
    '''
    nodes=['Total']
    nodes=nodes+list(df.columns)+list(df.index)
    nodes=pd.Series(nodes)
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


def table(data,code):
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
        if 'weight' in code:
            fw=pd.DataFrame(columns=[u'加权'],index=code['qlist'])
            w=pd.Series(code['weight'])
            for c in fo.columns:
                t=fo[c]
                t=t[w.index][t[w.index].notnull()]
                fw.loc[c,u'加权']=(t*w).sum()/t.sum()
            fw.rename(index=code['code_r'],inplace=True)
            result['fw']=fw
        fo.rename(columns=code['code_r'],index=code['code'],inplace=True)
        fop=fo.copy()
        fop=fop/sample_len
        result['fop']=fop
        result['fo']=fo
    elif qtype == u'排序题':
        #提供综合统计和TOP1值统计
        # 其中综合的算法是当成单选题，给每个TOP分配和为1的权重
        topn=max([len(data[q][data[q].notnull()].unique()) for q in index])
        qsort=dict(zip([i+1 for i in range(topn)],[(topn-i)*2.0/(topn+1)/topn for i in range(topn)]))
        top1=data.applymap(lambda x:int(x==1))
        data.replace(qsort,inplace=True)
        t1=pd.DataFrame()
        t1['TOP1']=top1.sum()
        t1[u'综合']=data.sum()
        t1.sort_values(by=u'综合',ascending=False,inplace=True)
        t1.rename(index=code['code'],inplace=True)
        t=t1.copy()
        t=t/sample_len
        result['fop']=t
        result['fo']=t1
    else:
        result['fop']=None
        result['fo']=None
    return result   

def ntable(data,code):
    '''【后期将删除】
    单个题目描述统计
    code是data的编码，列数大于1
    返回两个数据：
    t1：默认的百分比表
    t2：原始频数表，且添加了合计项
    '''
    # 单选题
    qtype=code['qtype']
    index=code['qlist']
    data=pd.DataFrame(data)
    sample_len=data[code['qlist']].notnull().T.any().sum()
    if qtype == u'单选题':
        t1=data.iloc[:,0].value_counts()
        t1.sort_values(ascending=False,inplace=True)
        t=t1.copy()
        t=t/t.sum()*1.0
        t[u'合计']=t.sum()
        t1[u'合计']=t1.sum()
        t.rename(index=code['code'],inplace=True)
        t1.rename(index=code['code'],inplace=True)
        t.name=u'占比'
        t1.name=u'频数'        
        t=pd.DataFrame(t)
        t1=pd.DataFrame(t1)
    elif qtype == u'多选题':
        t1=data.sum()
        t1.sort_values(ascending=False,inplace=True)
        t1[u'合计']=t1.sum()
        t1.rename(index=code['code'],inplace=True)
        t=t1.copy()
        t=t/sample_len
        t.name=u'占比'
        t1.name=u'频数'   
        t=pd.DataFrame(t)
        t1=pd.DataFrame(t1)
    elif qtype == u'矩阵单选题':
        sample_len
        t1=pd.DataFrame(columns=code['qlist'],index=sorted(code['code']))
        for i in t1.columns:
            t1.loc[:,i]=data[i].value_counts()
        t1.rename(columns=code['code_r'],index=code['code'],inplace=True)
        t=t1.copy()
        t=t/sample_len
    elif qtype == u'排序题':
        #提供综合统计和TOP1值统计
        # 其中综合的算法是当成单选题，给每个TOP分配和为1的权重
        topn=max([len(data[q][data[q].notnull()].unique()) for q in index])
        qsort=dict(zip([i+1 for i in range(topn)],[(topn-i)*2.0/(topn+1)/topn for i in range(topn)]))
        top1=data.applymap(lambda x:int(x==1))
        data.replace(qsort,inplace=True)
        t1=pd.DataFrame()
        t1['TOP1']=top1.sum()
        t1[u'综合']=data.sum()
        t1.sort_values(by=u'综合',ascending=False,inplace=True)
        t1.rename(index=code['code'],inplace=True)
        t=t1.copy()
        t=t/sample_len
    else:
        t=None
        t1=None
    return (t,t1)

def crosstab(data_index,data_column,code_index=None,code_column=None,qtype=None):
    '''适用于问卷数据的交叉统计
    输入参数：
    data_index: 因变量，放在行中
    data_column:自变量，放在列中
    qtype: 给定两个数据的题目类型，若为字符串则给定data_index，若为列表，则给定两个的
    code_index: dict格式，指定data_index的编码等信息
    code_column: dict格式，指定data_column的编码等信息
    返回字典格式数据
    'fop'：默认的百分比表，行是data_index,列是data_column
    'fo'：原始频数表，且添加了总体项
    'fw': 加权平均值
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
    result={}
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
                fw.loc[c,u'加权']=(tmp*w).sum()/tmp.sum()
            fo1=data_index.sum()[w.index][data_index.sum()[w.index].notnull()]
            fw.loc[u'总体',u'加权']=(fo1*w).sum()/fo1.sum()
            result['fw']=fw
        t[u'总体']=data_index.sum()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        for i in t.columns:
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
        data_index.replace(qsort,inplace=True)
        t=pd.DataFrame(np.dot(data_index.fillna(0).T,data_column.fillna(0)))
        t.rename(index=dict(zip(range(R),index_list)),columns=dict(zip(range(C),columns_list)),inplace=True)
        t[u'总体']=data_index.sum()
        t.sort_values([u'总体'],ascending=False,inplace=True)
        t1=t.copy()
        for i in t.columns:
            t.loc[:,i]=t.loc[:,i]/column_freq[i]
        result['fop']=t
        result['fo']=t1              
    else:
        result['fop']=None
        result['fo']=None
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
    
    

def qtable(data,code,q1=None,q2=None):
    '''简易频数统计函数
    # 单个变量的频数统计
    qtable(data,code,'Q1')
    # 两个变量的交叉统计
    qtable(data,code,'Q1','Q2')
    
    '''
    if q2 is None:
        result=table(data[code[q1]['qlist']],code[q1])
    else:
        result=crosstab(data[code[q1]['qlist']],data[code[q2]['qlist']],code[q1],code[q2])
    return result

def contingency(fo,alpha=0.05):
    ''' 列联表分析：(观察频数表分析)
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
    .'p-value':
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
    R,C=fo.shape
    if u'总体' in fo.columns:
        fo.drop([u'总体'],axis=1,inplace=True)
    if u'合计' in fo.index:
        fo.drop([u'合计'],axis=0,inplace=True)
    fe=fo.copy()
    N=fo.sum().sum()
    for i in fe.index:
        for j in fe.columns:
            fe.loc[i,j]=fe.loc[i,:].sum()*fe.loc[:,j].sum()/float(N)
    TGI=fo/fe
    TWI=fo-fe
    CHI=np.sqrt((fo-fe)**2/fe)*(TWI.applymap(lambda x: int(x>0))*2-1)
    PCHI=1/(1+np.exp(-1*CHI))
    cdata['FO']=fo
    cdata['FE']=fe
    cdata['TGI']=TGI
    cdata['TWI']=TWI
    cdata['CHI']=CHI
    cdata['PCHI']=PCHI

    # 显著性检验(独立性检验)
    significant={}
    significant['threshold']=stats.chi2.ppf(q=1-alpha,df=C-1)
    #threshold=math.ceil(R*C*0.2)# 期望频数和实际频数不得小于5
    # 去除行变量中行为0的列
    fo=fo[fo.sum(axis=1)>10]
    if (fo.shape[0]<=1) or (np.any(fo.sum()==0)) or (np.any(fo.sum(axis=1)==0)):
        significant['result']=-2
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
    for c in CHI.columns:
        tmp=['%s'%s for s in list(CHI.index[CHI[c]>summary['chi_mean']+summary['chi_std']])]
        if tmp:
            tmp1=u'{col}更喜欢{s}'.format(col=c,s=','.join(tmp))
            conclusion=conclusion+tmp1+'; \n'
    #conclusion=';'.join([u'{col}更喜欢{s}'.format(col=c,s=','.join(['%s'%s for s in \
    #list(CHI.index[CHI[c]>summary['chi_mean']+summary['chi_std']])])) for c in CHI.columns])
    if not conclusion:
        conclusion=u'没有找到显著的结论'
    summary['summary']=conclusion
    cdata['summary']=summary
    return cdata




def cross_chart(data,code,cross_class,filename=u'交叉分析', cross_qlist=None,\
delclass=None,plt_dstyle=None,cross_order=None, significance_test=False, \
reverse_display=False,total_display=True,max_column_chart=20,save_dstyle=None,\
template=None):

    '''使用帮助
    data: 问卷数据，包含交叉变量和所有的因变量
    code: 数据编码
    cross_class: 交叉变量，单选题或者多选题，例如：Q1
    filename：文件名,用于PPT和保存相关数据
    cross_list: 需要交叉分析的变量，缺省为code中的所有变量
    delclass: 交叉变量中需要删除的单个变量，缺省空
    plt_dstyle: 绘制图表需要用的数据类型，默认为百分比表，可以选择['TGI'、'CHI'、'TWI']等
    save_dstyle: 需要保存的数据类型，格式为列表。
    cross_order: 交叉变量中各个类别的顺序，可以缺少
    significance_test: 输出显著性校验结果，默认无
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
        cross_qlist.remove(cross_class)

    if significance_test:
        difference=dict(zip(cross_qlist,[-1]*len(cross_qlist)))
        difference[cross_class]=-2 #-2就代表了是交叉题目

    # =================基本数据获取==========================
    #交叉分析的样本数统一为交叉变量的样本数
    sample_len=data[code[cross_class]['qlist']].notnull().T.any().sum()


    # 交叉变量中每个类别的频数分布.
    if code[cross_class]['qtype'] == u'单选题':
        #data[cross_class].replace(code[cross_class]['code'],inplace=True)
        cross_class_freq=data[cross_class].value_counts()
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
    if template:
        prs=Presentation(template['path'])
        layouts=template['layouts']
    else:
        prs = Presentation()
        layouts=[0,5]
    if not os.path.exists('.\\out'):
        os.mkdir('.\\out')
    # 生成数据接口(因为exec&eval)
    Writer=pd.ExcelWriter('.\\out\\'+filename+u'_百分比表.xlsx')
    Writer_save={}
    if save_dstyle:
        for dstyle in save_dstyle:
            Writer_save[u'Writer_'+dstyle]=pd.ExcelWriter('.\\out\\'+filename+u'_'+dstyle+'.xlsx')

    result={}#记录每道题的的统计数据
    # ================背景页=============================
    title=u'背景说明(Powered by Python)'
    summary=u'交叉题目为'+cross_class+u': '+code[cross_class]['content']
    summary=summary+'\n'+u'各类别样本量如下：'
    plot_table(prs,cross_class_freq,title=title,summary=summary,layouts=layouts)
    data_column=data[code[cross_class]['qlist']]
    for qq in cross_qlist:
        # 遍历所有题目
        qtitle=code[qq]['content']
        qlist=code[qq]['qlist']
        qtype=code[qq]['qtype']
        data_index=data[qlist]

        sample_len=data_column.iloc[list(data_index.notnull().T.any()),:].notnull().T.any().sum()
        summary=None
        if qtype not in [u'单选题',u'多选题',u'排序题',u'矩阵单选题']:
            continue
        # 交叉统计
        if reverse_display:
            result_t=crosstab(data_column,data_index,code_index=code[cross_class],code_column=code[qq])
        else:
            result_t=crosstab(data_index,data_column,code_index=code[qq],code_column=code[cross_class])
        t=result_t['fop']
        t1=result_t['fo']
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
        if 'code_order' in code[qq]:
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
        t.fillna(0,inplace=True)
        t1.fillna(0,inplace=True)
        t2=pd.concat([t,t1],axis=1)

        # =======保存到Excel中========
        t2.to_excel(Writer,qq,index_label=qq,float_format='%.3f')

        #列联表分析
        cdata=contingency(t1,alpha=0.05)
        result[qq]=cdata
        summary=cdata['summary']['summary']
        if plt_dstyle:
            plt_data=cdata[plt_dstyle]
        elif qtype in [u'单选题',u'多选题']:
            plt_data=t*100
        else:
            plt_data=t.copy()

        # 保存各个指标的数据
        if save_dstyle:
            for dstyle in save_dstyle:
                cdata[dstyle].to_excel(Writer_save[u'Writer_'+dstyle],qq,index_label=qq,float_format='%.2f')

        
        # ========================【特殊题型处理区】================================
        if 'fw' in result_t:
            plt_data=result_t['fw']
            if cross_order and (not reverse_display):
                if u'总体' not in cross_order:
                    cross_order=cross_order+[u'总体']
                cross_order=[q for q in cross_order if q in plt_data.index]
                plt_data=pd.DataFrame(plt_data,index=cross_order)
        
        # 绘制PPT
        title=qq+': '+qtitle
        if not summary:
            summary=u'这里是结论区域.'
        footnote=u'显著性检验结果为{result},数据来源于{qq},样本N={sample_len}'.format(result=cdata['significant']['result'],qq=qq,sample_len=sample_len)
        if (not total_display) and (u'总体' in plt_data.columns):
            plt_data.drop([u'总体'],axis=1,inplace=True)
        if len(plt_data)>max_column_chart:
            plot_chart(prs,plt_data,'BAR_CLUSTERED',title=title,summary=summary,\
            footnote=footnote,layouts=layouts)
        else:
            plot_chart(prs,plt_data,'COLUMN_CLUSTERED',title=title,summary=summary,\
            footnote=footnote,layouts=layouts)




    '''
    # ==============小结页=====================
    difference=pd.Series(difference,index=total_qlist_0)
    '''

    # ========================文件生成和导出======================
    #difference.to_csv('.\\out\\'+filename+u'_显著性检验.csv',encoding='gbk')
    prs.save('.\\out\\'+filename+u'.pptx')
    Writer.save()
    if save_dstyle:
        for dstyle in save_dstyle:
            Writer_save[u'Writer_'+dstyle].save()
    return result


def summary_chart(data,code,filename=u'描述统计报告', summary_qlist=None,\
significance_test=False, max_column_chart=20,template=None):

    # ===================参数预处理=======================
    if not summary_qlist:
        try:
            summary_qlist=list(sorted(code,key=lambda c: int(re.findall('\d+',c)[0])))
        except:
            summary_qlist=list(code.keys())

    '''
    if significance_test:
        difference=dict(zip(cross_qlist,[-1]*len(cross_qlist)))
        difference[cross_class]=-2 #-2就代表了是交叉题目
    '''

    # =================基本数据获取==========================
    #统一的有效样本，各个题目可能有不能的样本数
    sample_len=len(data)

    # ================I/O接口=============================
    if template:
        prs=Presentation(template['path'])
        layouts=template['layouts']
    else:
        prs = Presentation()
        layouts=[0,5]
    if not os.path.exists('.\\out'):
        os.mkdir('.\\out')
    Writer=pd.ExcelWriter('.\\out\\'+filename+'.xlsx')
    result={}#记录每道题的过程数据
    # ================背景页=============================
    title=u'背景说明(Powered by Python)'
    summary=u'有效样本为%d'%sample_len
    plot_textbox(prs,title=title,summary=summary,layouts=layouts)


    for qq in summary_qlist:
        '''
        特殊题型处理
        整体满意度题：后期归为数值类题型
        '''

        qtitle=code[qq]['content']
        qlist=code[qq]['qlist']
        qtype=code[qq]['qtype']
        sample_len_qq=data[code[qq]['qlist']].notnull().T.any().sum()
        if qtype not in [u'单选题',u'多选题',u'排序题',u'矩阵单选题']:
            continue
        result_t=table(data[qlist],code=code[qq])
        t=result_t['fop']
        t1=result_t['fo']

        # =======数据修正==============
        if 'code_order' in code[qq]:
            code_order=code[qq]['code_order']
            #code_order=[q for q in code_order if q in t.index]
            if u'合计' in t.index:
                code_order=code_order+[u'合计']
            t=pd.DataFrame(t,index=code_order)
            t1=pd.DataFrame(t1,index=code_order)
        t.fillna(0,inplace=True)
        t1.fillna(0,inplace=True)
        t2=pd.concat([t,t1],axis=1)

        # =======保存到Excel中========
        t2.to_excel(Writer,qq,index_label=qq,float_format='%.3f')

        '''显著性分析[暂缺]
        cc=contingency(t,col_dis=None,row_dis=None,alpha=0.05)
        '''



        '''
        # ========================【特殊题型处理区】================================
        if ('name' in code[qq].keys()) and code[qq]['name'] in [u'满意度','satisfaction']:
            title=u'整体满意度'
        '''
        # 数据再加工
        if qtype in [u'单选题',u'多选题']:
            plt_data=t*100
        else:
            plt_data=t.copy()
        if u'合计' in plt_data.index:
            plt_data.drop([u'合计'],axis=0,inplace=True)
        result[qq]=plt_data
        title=qq+': '+qtitle
        if (qtype in [u'单选题']) and 'fw' in result_t:
            summary=u'这里是结论区域, 平均值为%s'%result_t['fw']
        else:
            summary=u'这里是结论区域.'
        footnote=u'数据来源于%s,样本N=%d'%(qq,sample_len_qq)
        format1={'value_axis.tick_labels.number_format':'\'0"%"\'',\
        'value_axis.tick_labels.font.size':Pt(10),\
        }
        if len(t)>max_column_chart:
            plot_chart(prs,plt_data,'BAR_CLUSTERED',title=title,summary=summary,\
            footnote=footnote,chart_format=format1,layouts=layouts)
        elif len(t)>3:
            plot_chart(prs,plt_data,'COLUMN_CLUSTERED',title=title,summary=summary,\
            footnote=footnote,chart_format=format1,layouts=layouts)
        else:
            plot_chart(prs,plt_data,'PIE',title=title,summary=summary,\
            footnote=footnote,layouts=layouts)




    '''
    # ==============小结页=====================
    difference=pd.Series(difference,index=total_qlist_0)
    '''

    # ========================文件生成和导出======================
    #difference.to_csv('.\\out\\'+filename+u'_显著性检验.csv',encoding='gbk')
    prs.save('.\\out\\'+filename+u'.pptx')
    Writer.save()
    return result




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
