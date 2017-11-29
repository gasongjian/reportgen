# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:39:10 2017

@author: JSong
"""

import os
import sys


_thisdir = os.path.realpath(os.path.split(__file__)[0])


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
        if isinstance(suffix,list) and not(os.path.splitext(f)[1][1:] in suffix):
            element_path=None
        else:
            element_path=os.path.join(_thisdir,dir_name,f)
    return element_path


# default pptx template
template_pptx=_get_element_path('template',suffix=['pptx'])
#template='template.pptx'


# default font of chinese  
font_path=_get_element_path('font',suffix=['ttf','ttc'])
if font_path is None:
    if sys.platform.startswith('win') and os.path.exists('C:\\windows\\fonts\\msyh.ttc'):
        font_path='C:\\windows\\fonts\\msyh.ttc'
        

# PPT图表中的数字位数
number_format_chart_1='0"%"'
number_format_chart_2='0.00'

# PPT图表中坐标轴的数字标签格式
number_format_chart_label='0"%"'

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






