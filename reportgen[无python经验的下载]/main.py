# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:16:07 2017

@author: gason
"""

# simple_run.py

# -*- coding: utf-8 -*-

import report as rpt
import re
import os
import time

mytemplate={'path':'template.pptx','layouts':[2,0]}

print('=='*15+'[reportgen 工具包]'+'=='*15)

#==================================================================
while 1:
    #print('=' * 70)
    try:
        command = input('''
==========一、数据准备=======.
1. 从问卷星导入数据并编码.
2.从问卷网导入数据并编码.
3.直接导入已编码好的数据.
4.继续下一步：报告生成
请输入相应的序号
''')
            
        if command in ['4','exit']:
            #print('开始下一步...')
            break
        if command=='1':
            print('准备导入问卷星数据，请确保“.\data\”文件夹下有按序号和按文本数据(如100_100_0.xls、100_100_2.xls).')
            try:
                data,code=rpt.wenjuanxing()
            except Exception as e:
                print(e)
                print('问卷星数据导入失败, 请检查.')
                continue
            cross_qlist=list(sorted(code,key=lambda c: int(re.findall('\d+',c)[0])))
            print('将题目进行编码......\n')
            for k in cross_qlist:
                print('{key}:  {c}'.format(key=k,c=code[k]['content']))
                time.sleep(0.1)
            rpt.save_code(code,'code.xlsx')
            rpt.save_data(data,'data.xlsx')
            rpt.save_data(data,'data_readable.xlsx',code)
            break
        if command=='2':
            print('准备导入问卷网数据，请确保“.\data\”文件夹下有按序号、按文本和code数据.')
            try:
                data,code=rpt.wenjuanwang()
            except Exception as e:
                print(e)
                print('问卷网数据导入失败, 请检查.')
                continue
            cross_qlist=list(sorted(code,key=lambda c: int(re.findall('\d+',c)[0])))
            print('将题目进行编码......\n')
            for k in cross_qlist:
                print('{key}:  {c}'.format(key=k,c=code[k]['content']))
                time.sleep(0.1)
            rpt.save_code(code,'code.xlsx')
            rpt.save_data(data,'data.xlsx')
            rpt.save_data(data,'data_readable.xlsx',code)
            break               
        if command=='3':
            data_name=input('请输入数据的文件名，缺省为 data.xlsx. 请输入:')
            if not data_name:
                data_name='data.xlsx'
            try:
                data=rpt.read_data(data_name)
                print('已成功导入data.')
            except Exception as e:
                print(e)
                print('data导入失败, 请检查')
                continue
            code_name=input('请输入code的文件名，缺省为 code.xlsx. 请输入:')
            if not code_name:
                code_name='code.xlsx'
            try:
                code=rpt.read_code(code_name)
                print('已成功导入code.')
            except Exception as e:
                print(e)
                print('code导入失败, 请检查')
                continue
            cross_qlist=list(sorted(code,key=lambda c: int(re.findall('\d+',c)[0])))
            print('题目编码情况如下......\n')
            for k in cross_qlist:
                print('{key}:  {c}'.format(key=k,c=code[k]['content']))
                time.sleep(0.1)
            break
    except Exception as e:
        print(e)
        print('错误..')

#=======================================================================
while 1:
    print('=' * 70)
    try:
        command = input('''
==========二、报告生成=======.
1. 描述统计报告(无交叉分析).
2. 交叉分析报告.
3. 退出程序(也可以输入exit).
请输入相应的序号
''')
            
        if command in ['3','exit']:
            print('本工具包由JSong开发, 谢谢使用，再见..')
            break
        if command=='1':
            filename=input('请输入需要保存的文件名,缺省为调研报告初稿: ')
            if not filename:
                filename=u'调研报告初稿'
            rpt.summary_chart(data,code,filename=filename,template=mytemplate);
            print('\n 报告已生成: '+os.path.join(os.getcwd(),'out',filename+'.pptx'))
        if command=='2':
            qq=input('请输入需要交叉分析的变量(例如: Q1): ')
            if qq in code:
                print('您输入的是%s: %s'%(qq,code[qq]['content']))
            else:
                print('没有找到您输入的题目,请返回重新输入.')
                continue
            if code[qq]['qtype'] not in ['单选题','多选题']:
                print('您选择的题目类型不是单选题或者多选题，本脚本暂时无法支持，请重新选择！')
                continue
            filename=qq+'_差异分析'
            save_dstyle=['FO','TGI','CHI']
            print('脚本正在努力生成报告中，请耐心等待........')
            try:
                rpt.cross_chart(data,code,qq,filename=filename,save_dstyle=save_dstyle,template=mytemplate);
                print('\n 报告已生成: '+os.path.join(os.getcwd(),'out',filename+'.pptx'))
            except Exception as e:
                print(e)
                print('报告生成过程出现错误，请重新检查数据和编码.')
                continue                    
    except Exception as e:
        print(e)
        print('错误..')

       