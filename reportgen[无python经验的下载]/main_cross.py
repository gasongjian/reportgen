# -*- coding: utf-8 -*-

import report as rpt
import re
import time

mytemplate={'path':'template.pptx','layouts':[1,0]}
data,code=rpt.wenjuanxing()

cross_qlist=list(sorted(code,key=lambda c: int(re.findall('\d+',c)[0])))

print('=='*15+'[reportgen 工具包]'+'=='*15)
print('将题目进行编码......\n')
for k in cross_qlist:
    print('{key}:  {c}'.format(key=k,c=code[k]['content']))
    time.sleep(0.1)

while 1:
    print('=' * 70)
    try:
        command = input(
            '本脚本用于交叉分析报告的一键生成.\n1. 请输入需要交叉分析的变量(例如: Q1) \n2. 输入exit或quit退出程序\n请输入:')
        if command == 'exit' or command == 'quit':
            print('程序结束...')
            break       
        print('您输入的是%s: %s'%(command,code[command]['content']))
        if code[command]['qtype'] not in ['单选题','多选题']:
            print('您选择的题目类型不是单选题或者多选题，本脚本暂时无法支持，请重新选择！')
            continue
        filename=command+'_差异分析'
        save_dstyle=['FO','TGI','TWI','CHI']
        print('脚本正在努力生成报告中，请耐心等待........')
        rpt.cross_chart(data,code,command,filename=filename,save_dstyle=save_dstyle,template=mytemplate);
        print('已完成交叉分析报告和相关数据的生成, 请检查文件夹：".\out\"')
    except Exception as e:
        print(e)
        print('错误..')

