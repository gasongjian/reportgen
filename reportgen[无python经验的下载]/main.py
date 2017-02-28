# -*- coding: utf-8 -*-
import report as rpt

print('=='*15+'[reportgen 工具包]'+'=='*15)
mytemplate={'path':'template.pptx','layouts':[1,0]}
data,code=rpt.wenjuanxing()
rpt.save_data(data,'Data_Original.xlsx')
rpt.save_data(data,'Data_Readable.xlsx',code)
rpt.save_code(code,'code.xlsx')
print('1、数据编码成功, 并输出为:\nData_Original.xlsx.\nData_Readable.xlsx.\ncode.xlsx')
# 描述统计报告生成
rpt.summary_chart(data,code,filename=u'调研报告初稿',template=mytemplate);
print('\n2、报告已生成, 请检查文件夹：".\out\"')
