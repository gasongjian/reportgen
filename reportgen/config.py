# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:39:10 2017

@author: JSong
"""

import os
import sys
import pandas as pd


_thisdir = os.path.realpath(os.path.split(__file__)[0])

__all__=['template_pptx',
         'font_path',
         'chart_type_list',
         'number_format_data',
         'number_format_tick',
         'font_default_size',
         'summary_loc',
         'chart_loc']

def _get_element_path(dir_name,suffix=None):
    if not(os.path.exists(os.path.join(_thisdir,dir_name))):
        element_path=None
        return element_path
    element_path=None
    filelist=os.listdir(os.path.join(_thisdir,dir_name))
    if isinstance(suffix,str):
        suffix=[suffix]
    elif suffix is not None:
        suffix=list(suffix)
    for f in filelist:
        if isinstance(suffix,list) and os.path.splitext(f)[1][1:] in suffix:
            element_path=os.path.join(_thisdir,dir_name,f)
    return element_path


# default pptx template
template_pptx=_get_element_path('template',suffix=['pptx'])
#template='template.pptx'


# default font of chinese
font_path=_get_element_path('font',suffix=['ttf','ttc'])
if font_path is None:
    if sys.platform.startswith('win'):
        #font_path='C:\\windows\\fonts\\msyh.ttc'
        fontlist=['calibri.ttf','simfang.ttf','simkai.ttf','simhei.ttf','simsun.ttc','msyh.ttf','MSYH.TTC','msyh.ttc']
        for f in fontlist:
            if os.path.exists(os.path.join('C:\\windows\\fonts\\',f)):
                font_path=os.path.join('C:\\windows\\fonts\\',f)

chart_type_list={\
"COLUMN_CLUSTERED":['柱状图','ChartData','pptx'],\
"BAR_CLUSTERED":['条形图','ChartData','pptx'],
'HIST':['分布图,KDE','XChartData','matplotlib']}
chart_type_list=pd.DataFrame(chart_type_list)


# PPT图表中的数字位数
number_format_data='0"%"'

# PPT图表中坐标轴的数字标签格式
number_format_tick='0"%"'

#  默认字体大小
'''
Pt(8):101600,  Pt(10):127000,  Pt(12):152400,  Pt(14):177800
Pt(16):203200,  Pt(18):228600,  Pt(20):254000,  Pt(22):279400
Pt(24):304800,  Pt(26):330200
'''
font_default_size=127000# Pt(10)


#  PPT中结论文本框所在的位置
# 四个值依次为left、top、width、height
summary_loc=[0.10,0.14,0.80,0.15]


#  PPT中结论文本框所在的位置
# 四个值依次为left、top、width、height
chart_loc=[0.10,0.30,0.80,0.60]
