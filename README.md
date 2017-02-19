# reportgen

> 问卷数据分析，支持一键生成描述统计和交叉分析报告。其中交叉分析报告支持提取简单的结论



## 安装

依赖环境： 
1. python科学计算所需的包，建议直接安装[anaconda](https://www.continuum.io/downloads)(兼容python2和python3)
2. `python-pptx`:  可以用"pip install python-pptx" 安装



## 快速上手

### 三行代码解决描述统计报告：

1. `调研报告初稿.pptx`: 针对每个题目描述统计，支持单选题、多选题、排序题、矩阵单选题等
2. `调研报告初稿.xlsx`: 生成每个题目的统计数据，包括频数和占比

```python
import report as rpt
#  数据编码和导入
data,code=rpt.wenjuanxing()#将问卷星的按文本数据和按序号数据放在“.\\data\\”中即可，也可以自定义路径
# 描述统计报告生成
rpt.summary_chart(data,code,filename=u'调研报告初稿')
```

### 四行代码解决交叉统计报告

1. `性别差异分析.pptx`: 考虑每个题目在性别上的差异
2. `性别差异分析_百分比.xlsx`:
3. `性别差异分析_FE.xlsx`:
4. `性别差异分析_TGI.xlsx`:
5. `性别差异分析_CHI.xlsx`:

```python
import report as rpt
#  数据编码和导入
data,code=rpt.wenjuanxing()
# 交叉统计报告生成(假设第一道题Q1是性别选择题)
save_dstyle=['FE','TGI','CHI']#自由选择需要保存的指标(FE:期望频数等)
rpt.cross_chart(data,code,cross_class='Q1',filename=u'性别差异分析',save_dstyle=save_dstyle)
```


## 问卷数据分析










