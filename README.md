# reportgen

> 问卷数据分析工具包，支持一键生成描述统计和交叉分析报告。其中交叉分析报告支持提取简单的结论



## 1、准备工作

依赖环境：

1. python科学计算所需的包，建议直接安装[anaconda](https://www.continuum.io/downloads)(强烈推荐使用python3版本)
2. 安装python包`python-pptx`:  在cmd中输入："pip install python-pptx" 
3. 安装report包: 下载report\\report.py, 然后放在工作目录即可(省心点可以直接扔进 C:\Anaconda3\Lib\site-packages 中，这样在任何地方都能使用该工具包啦)
4. 打开spyder

#### 备注

py2.7版本的pptx包对中文支持有 bug, 请按照如下方式修改

1. 打开文件 ".\\pptx\\chart\\xmlwriter.py"
2. 将大约1338行和1373行的 "escape(str(name))" 改为"escape(unicode(name))"

## 2、快速上手

### 2.1 三行代码解决描述统计报告：

```python
import report as rpt
#  数据编码和导入
# 300_300_0.xls是问卷星的按文本数据,300_300_2.xls是问卷星的按序号数据.
# 如果将他们放在“.\\data\\”中，则文件名可以缺省，即：`data,code=rpt.wenjuanxing()`
data,code=rpt.wenjuanxing(['300_300_0.xls','300_300_2.xls'])
# 描述统计报告生成
rpt.summary_chart(data,code,filename=u'调研报告初稿');
```
如上代码可以在.\\out\\文件夹下生成两个文件

1. `调研报告初稿.pptx`: 针对每个题目描述统计，支持单选题、多选题、排序题、矩阵单选题等
2. `调研报告初稿.xlsx`: 生成每个题目的统计数据，包括频数和占比

### 2.2 四行代码解决交叉统计报告


```python
import report as rpt
#  数据编码和导入
data,code=rpt.wenjuanxing()
# 交叉统计报告生成(假设第一道题Q1是性别选择题)
save_dstyle=['FE','TGI','CHI']#自由选择需要保存的指标(FE:期望频数等)
rpt.cross_chart(data,code,cross_class='Q1',filename=u'性别差异分析',save_dstyle=save_dstyle);
```
如上代码可以在.\\out\\文件夹下生成5个文件

1. `性别差异分析.pptx`: 考虑每个题目在性别上的差异
2. `性别差异分析_百分比.xlsx`:
3. `性别差异分析_FE.xlsx`:
4. `性别差异分析_TGI.xlsx`:
5. `性别差异分析_CHI.xlsx`:


=========================
=========================


## 3、工具包入门

### 3.1 数据编码和预处理

问卷数据涉及到各种题型，包括单选题、多选题、填空题、矩阵多选题、排序题等等。不管是
频数统计还是交叉分析，单选题都很好处理。但其他题目就相对复杂的多，比如单选题和多选题
之间的交叉统计，多选题和多选题之间的交叉统计。

为了区分题目类型和统计处理方法，本工具包统一使用一种数据类型（或者说编码方式）：

1、按序号编码的数据(csv、xlsx等都可以)，示例如下：
   

Q1|Q2|Q3_A1|Q3_A2|Q3_A3|Q3_A4|
-----
1|2|1|0|1|0


2、 编码文件（json格式）, 给定每道题的题号、序号编码等内容，示例：

```json
code={'Q1':{
    'content':'性别',
    'code':{
        1:'男',
        2:'女'
    }
}
```

对于直接从问卷网和问卷星上下载的源数据，本工具包支持自动编码

#### 3.1.1 问卷网

问卷网上直接下载下来的数据文件有3个，分别是`all`()、`all`() 和`code.xlsx`


### 3.2 描述统计

### 3.3 交叉统计

### 3.4 列联合表分析

### 3.5 PPT生成

#### 3.5.1 模板的使用









