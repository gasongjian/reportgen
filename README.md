# reportgen

> 问卷数据分析工具包，支持一键生成描述统计和交叉分析报告。其中交叉分析报告支持提取简单的结论



## 1、准备工作

依赖环境：

1. python科学计算所需的包，建议直接安装[anaconda](https://www.continuum.io/downloads)(强烈推荐使用python3版本)
2. 安装python包`python-pptx`:  在cmd中输入："pip install python-pptx" 
3. 安装report包: 下载report\\report.py, 然后放在工作目录即可(省心点可以直接扔进 C:\Anaconda3\Lib\site-packages 中，这样在任何地方都能使用该工具包啦)


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

![描述统计报告](https://github.com/gasongjian/reportgen/tree/master/report/demo/demo1.png)

### 2.2 四行代码解决交叉统计报告


```python
import report as rpt
#  数据编码和导入
data,code=rpt.wenjuanxing()
# 交叉统计报告生成(假设第一道题Q1是性别选择题)
rpt.cross_chart(data,code,cross_class='Q1',filename=u'性别差异分析');
```
如上代码可以在.\\out\\文件夹下生成5个文件

1. `性别差异分析.pptx`: 考虑每个题目在性别上的差异
2. `性别差异分析_百分比.xlsx`:
3. `性别差异分析_FE.xlsx`:
4. `性别差异分析_TGI.xlsx`:
5. `性别差异分析_CHI.xlsx`:
![交叉分析报告](https://github.com/gasongjian/reportgen/tree/master/report/demo/demo2.png)

### 2.3 常用函数

```pyton
import report as rpt
# 文件I/O 
data=rpt.read_data(filename)
code=rpt.read_code(filename)
rpt.save_data(data,filename,code)
rpt.save_code(code,filename)
data,code=rpt.wenjuanxing(filepath)# 编码问卷星的数据
data,code=rpt.wenjuanwang(filepath)# 编码问卷网的数据
# 数据统计函数
t,t1=rpt.qtable(data,code,'Q1')# 单变量频数统计
t,t1=rpt.qtable(data,code,'Q1','Q2')# 双变量交叉统计
# 数据分析函数
cdata=rpt.contingency(fo)# 列联表分析
rpt.gof_test(fo,fe)# 拟合优度检验
rpt.chi2_test(fo,fe)# 卡方检验
rpt.binomial_interval(p,n)# 计算比率的置信区间
# 自动描述统计报告
'''
summary_qlist: 例如['Q1','Q2'],需要分析的问卷题目列表，缺省为code中所有的关键词
template: 例如{'path':'mytemplate.pptx','layouts':[1,2]}, 缺省为pptx自带的模板

'''
rpt.summary_chart(data,code,filename=u'描述统计报告', summary_qlist=None,\
max_column_chart=20,template=None)

# 自动交叉统计报告
'''
cross_class: 需要交叉分析的题目，如：'Q1'
cross_qlist: 例如['Q1','Q2'],需要分析的问卷题目列表，缺省为code中所有的关键词
plt_dstyle: 绘制在ppt上使用的数据格式，缺省为百分比表，可以选择'TGI'等
save_dstyle: 需要保存的数据，例如:['TGI','FO','TWI','CHI']
template: 例如{'path':'mytemplate.pptx','layouts':[1,2]}, 缺省为pptx自带的模板

'''
rpt.cross_chart(data,code,cross_class,filename=u'交叉分析', cross_qlist=None,\
delclass=None,plt_dstyle=None,cross_order=None, significance_test=False, \
reverse_display=False,total_display=True,max_column_chart=20,save_dstyle=None,\
template=None):
```


=========================
=========================


## 3、工具包教程

### 3.1 问卷数据的格式

问卷数据可以来源于各大问卷网站，如问卷星、问卷网等，也可以来源于用户手动填写后再人工录入的文件。但不管怎样，常见的题型都是：单选题、多选题、填空题、矩阵多选题、排序题等等。在数据分析的过程中，我们一般喜欢用数字来存储用户的选择，比如用1来代表18-24岁，用2代表25-29岁。这样处理的目的不仅仅是简介，更多的是因为一些复杂的分析算法必须用数字，例如相关性分析、聚类、关联分析等

接下来，我们将分题型来讨论问卷数据的存储方式。

首先是单选题，这个比较简单，我们可以用
之间的交叉统计，多选题和多选题之间的交叉统计。

为了区分题目类型和统计处理方法，本工具包统一使用一种数据类型（或者说编码方式）：

1、按序号编码的数据(csv、xlsx等都可以)，示例如下：
  

|Q1|Q2|Q3_A1|Q3_A2|Q3_A3|Q3_A4|
|:----:|:---:|:----:|:----:|:---:|:----:|
|1|1|1|0|1|0|
|1|2|0|0|1|0|
|1|1|1|0|0|1|
|2|3|0|1|1|0|
|1|2|1|0|1|0|
|1|4|0|1|0|1|
|2|2|1|0|1|0|
|1|1|0|1|0|1|
|2|2|1|0|1|0|


2、 编码文件（json格式）, 给定每道题的题号、序号编码等内容，示例：

```python
code={'Q1':{
    'content':'性别',
    'code':{
        1:'男',
        2:'女'
    }
    'qtype':'单选题',
    'qlist':['Q1']
},
'Q2':{
    'content':'年龄',
    'code':{
        1:'17岁以下',
        2:'18-25岁',
        3:'26-35岁',
        4:'36-46岁'
    },
    'qtype':'单选题',
    'qlist':['Q2']
},
'Q3':{
    'content':'爱好',
    'code':{
        'Q3_A1':'17岁以下',
        'Q3_A2':'18-25岁',
        'Q3_A3':'26-35岁',
        'Q3_A4':'36-46岁'
    },
    'qtype':'多选题',
    'qlist':['Q3_A1','Q3_A2','Q3_A3','Q3_A4']
}
}
# 其中为了便于修改code编码，本工具包提供了两个json于xlsx之间的相互转换函数
# rpt.read_code('code.xlsx')同样可以返回字典格式的code
```



### 3.2 数据导入
对于处理好的数据，可以用如下方式导入：

```python
data=rpt.read_data('Data_Original.xlsx')
code=rpt.read_code('code.xlsx')
```

对于从网站上下载的原始数据，可以使用如下方式导入:

```python
# 问卷星 
data,code=rpt.wenjuanxing(['320_320_0.xls','320_320_2.xls'])
# 参数如果缺省，函数会自动在工作目录的 `.\\data\\`文件夹中寻找

# 问卷网  
data,code=rpt.wenjuanwang(['All_Data_Readable.csv','All_Data_Original.csv','code.csv'])
# 参数如果缺省，函数会自动在工作目录的 `.\\data\\`文件夹中寻找

```

待写.....

### 3.3 数据预处理


### 3.4 数据分析

### 3.5 报告生成













