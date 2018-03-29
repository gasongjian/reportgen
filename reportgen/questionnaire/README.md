# reportgen v 0.1.8
-------------------

## **问卷模块** :问卷类型的数据分析

------------------

问卷数据涉及到各种题型，包括单选题、多选题、填空题、矩阵多选题、排序题等等。不管是频数统计还是交叉分析，单选题都很好处理。但其他题目就相对复杂的多，比如单选题和多选题之间的交叉统计，多选题和多选题之间的交叉统计。

为了区分题目类型和统计处理方法，本工具包统一使用新型的数据类型（或者说编码方式）。在这种类型中，每一份问卷都有两个文件，data 和 code ,它们的含义如下：

- 1)、data：按序号编码的数据(csv、xlsx等都可以)，示例如下：

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

- 2)、code：编码文件（ json格式，就是 python中的字典类型）, 给定每道题的题号、序号编码等内容，
    每一个题目都有如下字段：

        - content: 题目内容
        - code:题目对应的编码
        - code_r: 题目对应的编码(矩阵单选题专有)
        - qtype:题目类型，单选题、多选题、矩阵单选题、排序题、填空题等
        - qlist:该题的索引，如多选题的 ['Q1_A1','Q1_A2',..]
        - code_order: 非必须，题目类别的顺序，用于PPT报告的生成[一般后期添加]
        - name: 非必须，特殊题型的标注
        - weight:非必须，dict,每个选项的权重，用于如月收入等的平均数统计

    示例：

    ```json
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

##该工具包包含如下函数：

### 文件 IO

- `read_code`, 从本地读取code数据，支持excel文件和json文件
- `save_code`, 将code 保存为 xlsx 或json数据
- `load_data`, 支持打开文件窗口来选择问卷数据
- `read_data`, 读取本地的数据,自适应xlsx、csv等
- `save_data`, 将问卷数据（data和code）保存到本地
- `wenjuanwang`, 编码问卷网平台的问卷数据，输入为问卷网上下载的三个文件
- `wenjuanxing`, 编码问卷星平台的问卷数据，输入为问卷星网站上下载的两个xls文件（按选项序号和按选项文本）

### 数据处理
- `spec_rcode`: 对问卷中的一些特殊题型进行处理，如将城市题分类成省份、城市、城市级别等
- `dataText_to_code`:
- `dataCode_to_text`:
- `var_combine`: 见data_merge
- `data_merge`: 合并两份问卷数据，常见于多个推动渠道的问卷合并
- `clean_ftime`: 根据用户填写时间来筛选问卷，会根据填问卷累计时间曲线的拐点来给出剔除的时间点
- `data_auto_code`:
- `qdata_flatten`: 将问卷数据展平，便于将多份问卷数据存储在同一个数据库中

### 统计检验等
- `sample_size_cal`: 样本量计算公式
- `confidence_interval`: 置信区间计算公式
- `gof_test`: 拟合优度检验
- `chi2_test`: 卡方检验
- `fisher_exact`: 卡方检验，适用于观察频数过少的情形
- `anova`: 方差分析

### 数据分析
- `mca`: 对应分析，目前只支持两个变量
- `cluster`: 态度题的聚类分析，会根据轮廓系数自动选择最佳类别数
- `association_rules`: 关联分析，用于多选题的进一步分析

### 统计
- `contingency`: 列联表分析，统一给出列联表的各种数据，包含fo、fop、TGI等
- `qtable`: 单个题目的统计分析和两个题目的交叉分析，给出频数表和频率表

### 可视化
- `summary_chart`: 整体统计报告，针对每一道题，选择合适的图表进行展示，并输出为pptx文件
- `cross_chart`: 交叉分析报告，如能将年龄与每一道题目进行交叉分析，并输出为pptx文件
- `onekey_gen`: 综合上两个，一键生成
- `scorpion`: 生成一个表格，内含每个题目的相关统计信息
- `scatter`: 散点图绘制，不同于matplotlib的是，其能给每个点加文字标签
- `sankey`: 桑基图绘制，不画图，只提供 R 需要的数据
"""


## 一些实践：

数据在 .\\example\\datasets\\

```python
import reportgen.questionnaire as ques


# 导入问卷星数据
datapath=['.\\datasets\\[问卷星数据]800_800_0.xls','.\\datasets\\[问卷星数据]800_800_2.xls']
data,code=ques.wenjuanxing(datapath)

# 导出
ques.save_data(data,filename='data.xlsx')
ques.save_data(data,filename='data.xlsx',code=code)# 会将选项编码替换成文本
ques.save_code(code,filename='code.xlsx')


# 对单变量进行统计分析
result=ques.qtable(data,code,'Q1')
print(result['fo'])

# 两个变量的交叉分析
result=ques.qtable(data,code,'Q1','Q2')
print(result['fop'])

# 聚类分析，会在原数据上添加一列，类别题
#ques.cluster(data,code,'态度题')

# 在.\\out\\下 生成 pptx文件
ques.summary_chart(data,code,filename='整体统计报告');
ques.cross_chart(data,code,cross_class='Q4',filename='交叉分析报告_年龄');
ques.scorpion(data,code,filename='详细分析数据')
ques.onekey_gen(data,code,filename='reportgen 自动生成报告');
```
