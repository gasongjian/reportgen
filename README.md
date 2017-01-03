# reportgen

> matlab自动化报告(word)，请移步 [word(matlab)](https://github.com/gasongjian/reportgen/tree/master/word(matlab))。 
> 新出的这个项目是Python版本的，借用pandas和pptx实现问卷数据的自动化分析和报告生成



## 项目介绍

项目地址： python根据问卷半自动化生成问卷，见[reportgen](https://github.com/gasongjian/reportgen/tree/master/reportgen)


## PPT接口
```python
import report as rpt
from pptx import Presentation
import pandas as pd

prs=Presentation()
data=pd.DataFrame({'Q1':[0.1,0.4,0.2,0.3],'Q2':[0.4,0.2,0.15,0.25]})
t=pd.crosstab(data['Q1'],data['Q2'])
title=u'这是标题'
summary=u'这是一个测试的结论'
footnote=u'注：数据来源于Q1和Q2'
rpt.plot_chart(prs,t,'COLUMN_CLUSTERED',title=title,summary=summary,footnote=footnote)
```

## 描述统计



## 交叉分析






