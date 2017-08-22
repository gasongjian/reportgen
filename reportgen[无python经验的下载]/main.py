# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:16:07 2017

@author: gason
"""

# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.split(__file__)[0],'script'))

import re
import time
import pandas as pd
import report as rpt

import warnings
warnings.filterwarnings("ignore")

mytemplate='template.pptx'

print('=='*15+'[reportgen 工具包]'+'=='*15)

#==================================================================
while 1:
    #print('=' * 70)
    try:
        command = input('''
==========一、数据导入=======
1.导入问卷星数据并编码.
2.导入问卷网数据并编码.
3.导入已编码好的数据.
4.打开文件选择窗口选择（问卷星数据）.
请输入相应的序号:
''')
            
        if command in ['0','exit']:
            #print('开始下一步...')
            break
        if command=='1':
            print('准备导入问卷星数据，请确保“.\data\”文件夹下有按序号和按文本数据(如100_100_0.xls、100_100_2.xls).')
            filepath='.\\data'
            if os.path.isdir(filepath):
                filelist=os.listdir(filepath)
                wjx_data={}
                n1=n2=0
                for f in filelist:
                    s1=re.findall('(\d+_\d+)_0.xls',f)
                    s2=re.findall('(\d+_\d+)_2.xls',f)
                    if s1:
                        if s1[0] in wjx_data:
                            wjx_data[s1[0]][0]=f
                        else:
                            wjx_data[s1[0]]=[f,'']
                    if s2:
                        if s2[0] in wjx_data:
                            wjx_data[s2[0]][1]=f
                        else:
                            wjx_data[s2[0]]=['',f]
                tmp=[k for k in wjx_data if int(len(wjx_data[k][0])>0)+int(len(wjx_data[k][1])>0)==2]
                if len(tmp)==1:
                    # 刚好只识别出一组问卷星数据
                    filename1=os.path.join(filepath,tmp[0]+'_0.xls')
                    filename2=os.path.join(filepath,tmp[0]+'_2.xls')
                elif len(tmp)>1:
                    print('脚本识别出多组问卷星数据，请选择需要编码的数据：')
                    for i,k in enumerate(tmp):
                        print('{i}:  {k}'.format(i=i+1,k=k+'_0.xls/'+k+'_2.xls'))
                    ii=input('您选择的数据是(数据前的编码，如：1):')
                    if ii.isnumeric():
                        filename1=os.path.join(filepath,tmp[int(ii)-1]+'_0.xls')
                        filename2=os.path.join(filepath,tmp[int(ii)-1]+'_2.xls')
                    else:
                        print('您输入正确的编码.')
                        continue
                else:
                    print('在.\\data目录下没有找到任何的问卷星数据，请返回检查.')
                    continue          
            try:
                data,code=rpt.wenjuanxing([filename1,filename2])
                data,code=rpt.spec_rcode(data,code)
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
            print('\n编码完毕, 编码后的数据已经保存在本地为data.xlsx和code.xlsx. \n')
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
            print('编码完毕, 编码后的数据已经保存在本地为data.xlsx和code.xlsx. \n')
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

        if command=='4':
            import tkinter as tk
            from tkinter.filedialog import askopenfilenames
            tk.Tk().withdraw();
            print(u'请同时选择*按选项文本数据*和*按选项序号数据*') 
            initialdir = ".\\data"
            title =u"请选择按选项文本数据和按选项数据（同时选择两份数据）"
            filetypes = (("Excel files","*.xls;*.xlsx"),("all files","*.*"))
            filenames=[]
            while len(filenames)!=2:
                filenames=askopenfilenames(initialdir=initialdir,title=title,filetypes=filetypes)
                if len(filenames)!=2:
                    print('请重新选择，一共要选择两个文件.')
            dshape=[]
            for filepath in filenames:
                data=rpt.read_data(filepath)
                m,n=data.shape
                dshape=dshape+[m,n]
            if dshape[0]!=dshape[2]:
                print('选择的两组数据应该不是同一份问卷的，请返回检查.')
                continue
            if dshape[1]>dshape[3]:
                filename1=filenames[1]
                filename2=filenames[0]
            else:
                filename1=filenames[0]
                filename2=filenames[1]
            # 识别完后开始编码    
            try:
                data,code=rpt.wenjuanxing([filename1,filename2])
                data,code=rpt.spec_rcode(data,code)
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
            print('\n编码完毕, 编码后的数据已经保存在本地为data.xlsx和code.xlsx. \n')
            break            

    except Exception as e:
        print(e)
        print('错误..')

os.system('pause')


s='''
==========二、数据处理=======
1. 编码后的数据中，data.xlsx用于存放所有按序号数据，code.xlsx用于存放序号对应的文本内容以及报告生成所需变量

2. code中key[必须]：题号，content[必须]:题目内容，qtype[必须]:题目类型，qlist[必须]：题目在data.xlsx对应的位置，code_order:用于固定报告中选项的顺序,weight:用于求加权平均值，如NPS、满意度、模块满意度等.

3. 本脚本只提供简单的数据处理操作，因为没有界面体验很不好.大家可以在本地处理好数据后，重新导入编码后并修改好的数据。在这个过程中，如果涉及到选项序号的合并、修改等，请同步修改data和code两个文件，谢谢.

4. 如果需要修改数据，可以输入exit或者quit暂时退出本脚本，等数据修改完后再启动.
--------------------------------
'''
print(s)

command=input('按任意键进入[第二步：数据处理：]')

if command in ['0','exit','quit']:
    print('本工具包由JSong开发, 谢谢使用, 再见..')
    exit()


    
while 1:
    print('=' * 70)
    try:
        command = input('''
==========二、数据处理=======.
1. 数据筛选
2. 选项合并/修改
3. 态度题聚类分析
4. 因子分析（暂缺）
5. 特征工程（暂缺）
6. 将修改后的数据保存到本地（data.xlsx和code.xlsx）
7. 跳转到下一步
0. 退出程序(也可以输入exit或者quit)
请输入相应的序号:
''')

        if command in ['0','exit','quit']:
            print('本工具包由JSong开发, 谢谢使用, 再见..')
            break
        if command == '1':
            qq=input('请输入您需要进行[数据筛选]的题号(如Q1): ')
            qq=qq.upper()
            if qq in code:
                print('您输入的是%s: %s'%(qq,code[qq]['content']))
            else:
                print('没有找到您输入的题目,请返回重新输入.')
                continue
            print('\n该题的选项如下：')
            if code[qq]['qtype']=='单选题':
                for c,cvalue in code[qq]['code'].items():
                    print('  {}: {}'.format(c,cvalue))
                qlist=input('\n 请选择需要保留的选项（多个请用英文逗号分隔,如：1,2,5): ')
                if len(qlist)==0:
                    print('没有找到您输入的选项，请返回重新输入.')
                    continue
                qlist=re.sub('\s','',qlist)# 去掉空格
                qlist=re.sub('，',',',qlist)# 修正可能错误的逗号
                qlist=qlist.split(',')
                qlist=[int(qq) for qq in qlist]# 完善后的选项
                if not(set(qlist) < set(code['Q1']['code'].keys())):
                    print('您输入的选项与该题选项不匹配，请返回重新检查.')
                    continue
                data=data[data[code[qq]['qlist'][0]].isin(qlist)]
                code[qq]['code']={q:code[qq]['code'][q] for q in qlist}
                print('{}的筛选操作已完成.'.format(qq))
            elif code[qq]['qtype'] in ['多选题','排序题']:
                codeitems=list(code[qq]['code'].items())
                for ii in range(len(codeitems)):
                    print('  {}: {}'.format(ii+1,codeitems[ii][1]))
                qlist=input('\n 请选择需要保留的选项（多个请用英文逗号分隔,如：1,2,5): ')
                if len(qlist)==0:
                    print('没有找到您输入的选项，请返回重新输入.')
                    continue
                qlist=re.sub('\s','',qlist)# 去掉空格
                qlist=re.sub('，',',',qlist)# 修正可能错误的逗号
                qlist=qlist.split(',')
                qlist=[int(qq)-1 for qq in qlist]
                if not(set(qlist) < set(range(len(codeitems)))):
                    print('您输入的选项与该题选项不匹配，请返回重新检查.')
                    continue
                removelist=list(set(range(len(codeitems)))-set(qlist))#需要移除的选项
                removelist=[codeitems[i][0] for i in removelist]
                data=data.drop(removelist,axis=1)
                for c in removelist:
                    del code[qq]['code'][c]
                remainlist=[codeitems[i][0] for i in qlist]
                code[qq]['qlist']=[q for q in code[qq]['qlist'] if q in remainlist]
                print('{}的筛选操作已完成.'.format(qq))
            else:
                print('只支持筛选单选题、多选题、排序题，请返回重新选择.')
                continue
            continue
        if command == '2': 
            qq=input('请输入您需要进行[合并选项]的题号(如Q1): ')
            qq=qq.upper()
            if (qq in code) and (code[qq]['qtype']=='单选题'):
                print('您输入的是%s: %s'%(qq,code[qq]['content']))
            else:
                print('没有找到您输入的题目或者输入的不是单选题,请返回重新输入.')
                continue
            print('\n该题的选项如下：')
            for c,cvalue in code[qq]['code'].items():
                print('  {}: {}'.format(c,cvalue))
            print('\n您可以多次合并，直到所有选项都完成合并.在这个过程中，只要不输入并按回车键即可退出合并操作.\n')
            itemlist=[]# 记录合并前选项
            itemlist_new=[]# 记录合并后的选项
            itemname=[]# 记录合并后的选项名称
            while 1:
                itemlist0=input('请输入需要合并的选项(如:1,2): ')
                if len(itemlist0)==0:
                    break
                itemlist0=re.sub('\s','',itemlist0)# 去掉空格
                itemlist0=re.sub('，',',',itemlist0)# 修正可能错误的逗号
                itemlist0=itemlist0.split(',')
                itemlist0=[int(qq) for qq in itemlist0]
                itemname0=input('请输入合并后的新名称(不能和已有的名称重复): ')
                if len(itemname0)==0:
                    print('输入有误，请返回重新输入')
                    continue
                if itemname0 in code[qq]['code'].values():
                    print('和已有名称重复，请返回重新输入')
                    continue
                itemname0=itemname0.strip()
                itemlist=itemlist+itemlist0
                itemlist_new=itemlist_new+[itemlist0[0]]*len(itemlist0)                
                itemname=itemname+[itemname0]*len(itemlist0)
            # 替换data
            rcode=dict(zip(itemlist,itemlist_new))# 用户替换data
            data[code[qq]['qlist'][0]].replace(rcode,inplace=True)
            # 修改code
            additem=list(set(code[qq]['code'].keys())-set(itemlist))
            addname=[code[qq]['code'][c] for c in additem]
            code[qq]['code']=dict(zip(itemlist_new+additem,itemname+addname))
            print('{}的合并操作已完成.'.format(qq))
            continue
            
        if command == '3':
            s='''本脚本支持对态度题（题型为矩阵单选题）进行聚类分析，相关说明如下：
题型要求：矩阵单选题（量表题）
数据清洗：剔除了那些各选项的分值都几乎一样的样本
采用方法：K-means
最佳聚类数：根据轮廊系数自动选取，上限为7类')
结果说明：结果具有一定的随机性，仅供参考，另外可多次运行获取相对稳定的结论.
                    
            '''
            print(s)
            qq=input('请输入您需要进行[聚类分析]的题号(如Q1): ')
            qq=qq.upper()
            if (qq in code) and (code[qq]['qtype']=='矩阵单选题'):
                print('您输入的是%s: %s'%(qq,code[qq]['content']))
            else:
                print('您输入的题型不是矩阵单选题,请返回重新输入.')
                continue
            n_clusters=input('请输入分类个数(缺省则自动选择最佳聚类数): ')
            n_clusters=re.sub('\s','',n_clusters)# 去掉空格
            if len(n_clusters)==0:
                n_clusters='auto'
            else:
                n_clusters=int(n_clusters)
            data,code,para=rpt.cluster(data,code,qq,n_clusters=n_clusters)
            # 绘图
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            X=para['data']
            labels=para['labels']
            X_PCA = PCA(2).fit_transform(X)
            kwargs = dict(cmap = plt.cm.get_cmap('rainbow', 10),
                          edgecolor='none', alpha=0.6)
            plt.figure()
            plt.scatter(X_PCA[:, 0], X_PCA[:, 1], c=labels, **kwargs)
            filename='cluster_kmeans_{}_{}_pca.png'.format(qq,para[n_clusters])
            plt.savefig(filename,dpi=1200)
            print('聚类效果图已保存在本地：{}'.format(filename))
            try:
                from PIL import Image
                img=Image.open(filename)
                img.show()
            except:
                pass
            continue

        if command in ['4','5']:
            print('敬请期待.....')
            continue


        if command == '6':
            try:
                rpt.save_data(data)
                rpt.save_code(code)
                print('数据已经保存在本地')
            except :
                print('请关闭已经打开的文件：data.xlsx 或者 code.xlsx')
                print('关闭后返回重新选择')
                pass
            continue

        if command == '7':
            break
            
    except Exception as e:
        print(e)
        print('错误..')

#================三、数据预处理==============
#
#
#==========================================
while 1:
    print('=' * 70)
    try:
        command = input('''
==========三、报告生成=======.
x. 全自动一键生成
1. 整体统计报告自动生成
2. 交叉分析报告自动生成
3. 描述统计
4. 交叉分析
5. 对应分析
0. 退出程序(也可以输入exit或者quit)
请输入相应的序号:
''')
        if command in ['x','X']:
            filename=input('请输入需要保存的文件名,缺省为 reportgen报告自动生成: ')
            if not filename:
                filename=u'reportgen报告自动生成'
            print('请耐心等待，脚本正在马不停蹄地工作中......')
            rpt.onekey_gen(data,code,filename=filename,template=mytemplate);
            print('\n 所有报告已生成, 请检查文件夹：'+os.path.join(os.getcwd(),'out'))
            print('\n 开始生成*scorpion.xlsx*,请耐心等待')
            try:
                rpt.scorpion(data,code)
            except :
                print('脚本出现一些错误...')
            continue
        if command in ['0','exit','quit']:
            print('本工具包由JSong开发, 谢谢使用, 再见..')
            break
        if command=='1':
            filename=input('请输入需要保存的文件名,缺省为调研报告初稿: ')
            if not filename:
                filename=u'调研报告初稿'
            rpt.summary_chart(data,code,filename=filename,template=mytemplate);
            print('\n 报告已生成: '+os.path.join(os.getcwd(),'out',filename+'.pptx'))
            continue
        if command=='2':
            qq=input('请输入需要交叉分析的变量(例如: Q1): ')
            qq=qq.upper()
            if qq in code:
                print('您输入的是%s: %s'%(qq,code[qq]['content']))
            else:
                print('没有找到您输入的题目,请返回重新输入.')
                continue
            if code[qq]['qtype'] not in ['单选题','多选题']:
                print('您选择的题目类型不是单选题或者多选题，本脚本暂时无法支持，请重新输入！')
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
        if command=='3':
            qq=input('请输入需要统计的变量(例如: Q1): ')
            qq=qq.upper()
            if qq in code:
                print('您输入的是%s: %s'%(qq,code[qq]['content']))
            else:
                print('没有找到您输入的题目,请返回重新输入.')
                continue
            if code[qq]['qtype'] not in ['单选题','多选题','排序题','矩阵单选题']:
                print('您选择的题目类型本脚本暂时无法支持，请重新输入！')
                continue
            try:
                t=rpt.qtable(data,code,qq)
                if not(t['fo'] is None):
                    print('百分比表如下：')
                    print(t['fop'])
                    print('--'*10)
                    print('频数表如下:')
                    print(t['fo'])
            except Exception as e:
                print(e)
                print('脚本运行错误，请重新检查数据和编码.')
                continue

        if command=='4':
            qq1=input('请输入需要交叉分析的行变量，也是因变量(例如: Q1): ')
            if qq1 in code:
                print('您输入的是%s: %s'%(qq1,code[qq1]['content']))
            else:
                print('没有找到您输入的题目,请返回重新输入.')
                continue
            qq2=input('请输入需要交叉分析的列变量，也是自变量(例如: Q1): ')
            if qq2 in code:
                print('您输入的是%s: %s'%(qq2,code[qq2]['content']))
            else:
                print('没有找到您输入的题目,请返回重新输入.')
                continue
            if code[qq2]['qtype'] not in ['单选题','多选题']:
                print('您选择的自变量题目类型不是单选题或者多选题，本脚本暂时无法支持，请重新输入！')
                continue
            try:
                t=rpt.qtable(data,code,qq1,qq2)
                if not(t['fo'] is None):
                    print('百分比表如下：')
                    print(t['fop'])
                    print('--'*10)
                    print('频数表如下:')
                    print(t['fo'])
            except Exception as e:
                print(e)
                print('脚本运行错误，请重新检查数据和编码.')
                continue
        if command=='5':
            qq1=input('请输入需要对应分析的行变量，也是因变量(例如: Q1): ')
            if qq1 in code:
                print('您输入的是%s: %s'%(qq1,code[qq1]['content']))
            else:
                print('没有找到您输入的题目,请返回重新输入.')
                continue
            qq2=input('请输入需要交叉分析的列变量，也是自变量(例如: Q1): ')
            if qq2 in code:
                print('您输入的是%s: %s'%(qq2,code[qq2]['content']))
            else:
                print('没有找到您输入的题目,请返回重新输入.')
                continue
            if code[qq2]['qtype'] not in ['单选题','多选题']:
                print('您选择的自变量题目类型不是单选题或者多选题，本脚本暂时无法支持，请重新输入！')
                continue
            try:
                t=rpt.qtable(data,code,qq1,qq2)['fo']
                x,y,inertia=rpt.mca(t)
                title=u'对应分析图(信息量为{:.1f}%)'.format(inertia[1]*100)
                fig=rpt.scatter([x,y],title=title)
                filename='ca_'+qq1+'_'+qq2
                fig.savefig(filename+'.png',dpi=1200)
                w=pd.ExcelWriter(filename+'.xlsx')
                x.to_excel(w,startrow=0,index_label=True)
                y.to_excel(w,startrow=len(x)+2,index_label=True)
                w.save()
                print('该对应分析能解释{:.1f}%的信息,相应的图片和数据已保存为：'.format(inertia[1]*100))
                print('图片： '+filename+'.png')
                print('数据(可利用PPT的散点图绘制更漂亮的图表)：'+filename+'.xlsx')
                try:
                    from PIL import Image
                    img=Image.open(filename+'.png')
                    img.show()
                except:
                    pass
            except Exception as e:
                print(e)
                print('脚本运行错误，请重新检查数据和编码.')
                continue
        os.system('pause')
    except Exception as e:
        print(e)
        print('错误..')
        
        
        

       
