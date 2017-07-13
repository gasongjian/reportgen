from collections import defaultdict
import itertools
import pandas as pd

class apriori:
    def __init__(self, data, minSup, minConf):
        self.dataset = data
        self.transList = defaultdict(list)
        self.freqList = defaultdict(int)
        self.itemset = set()
        self.highSupportList = list()
        self.numItems = 0
        self.prepData()             # initialize the above collections

        self.F = defaultdict(list)

        self.minSup = minSup
        self.minConf = minConf

    def genAssociations(self):
        candidate = {}
        self.F[1] = self.firstPass(self.freqList, 1)
        k=2
        while len(self.F[k-1]) != 0:
            candidate[k] = self.candidateGen(self.F[k-1], k)
            # 循环每个样本,统计每个配对item的频数
            for t in self.transList.items():
                for c in candidate[k]:
                    if set(c).issubset(t[1]):
                        self.freqList[c] += 1
                #print(self.freqList[('Q11_A10,Q11_A7')])

            self.F[k] = self.prune(candidate[k], k)
            '''
            if k > 2:
                self.removeSkyline(k, k-1)
            '''
            k += 1

        return self.F

    def removeSkyline(self, k, kPrev):
        for item in self.F[k]:
            subsets = self.genSubsets(item)
            for subset in subsets:
                if subset in (self.F[kPrev]):
                    self.F[kPrev].remove(subset)                  
        subsets = self.genSubsets

    def prune(self, items, k):
        # 按照最小支持度剪枝
        f = []
        for item in items:
            count = self.freqList[item]
            support = self.support(count)
            if support >= .95:
                self.highSupportList.append(item)
            elif support >= self.minSup:
                f.append(item)

        return f
    # 计算候选的配对，item为配对集合，k为配对集的个数
    def candidateGen(self, items, k):
        candidate = []
        if k == 2:
            #candidate = [tuple(sorted([x, y])) for x in items for y in items if len((x, y)) == k and x != y]
            candidate = [tuple(sorted([items[i],items[j]])) for i in range(len(items)) for j in range(len(items)) if j>i]
        else:
            candidate = [tuple(set(x).union(y)) for x in items for y in items if len(set(x).union(y)) == k and x != y]
        
            for c in candidate:
                #类似于剪枝
                subsets = self.genSubsets(c)
                if any([ x not in items for x in subsets]):
                    candidate.remove(c)
        return set(candidate)

    def genSubsets(self, item):
        '''生成item的所有真子集
        genSubsets(('1','2')= [('1',), ('2',)]
        '''
        subsets = []
        for i in range(1,len(item)):
            subsets.extend(itertools.combinations(item, i))
        return subsets

    def genRules(self):
        '''生成关联规则
        subset --> rhs (其中rhs的长度为 1)
        '''
        F=self.genAssociations()
        H = []
        N=self.numItems
        freqList=[]
        for k, itemset in F.items():
            if k<2:
                continue
            for item in itemset:
                subsets = self.genSubsets(item)
                itemCount = self.freqList[item]
                support = itemCount/N
                freqList.append((item,support))
                for subset in subsets:
                    if len(subset)!=len(item)-1:
                        continue
                    rhs = self.difference(item, subset)
                    subCount=self.freqList[subset[0]] if len(subset)==1 else self.freqList[subset]
                    rhsCount=self.freqList[rhs[0]] if len(rhs)==1 else self.freqList[rhs]
                    if subCount==0 or rhsCount==0:
                        continue                    
                    confidence = self.confidence(subCount, itemCount)
                    lift=N*itemCount/rhsCount/subCount
                    if confidence >= self.minConf and lift >1:                         
                        H.append((subset, rhs, support, confidence,lift))
        freqList=pd.DataFrame(freqList,columns=['freq','sup'])
        freqList=freqList.sort_values(['sup'],ascending=False)
        freqList=freqList.reset_index(drop=True)

        if len(H)>0:
            H=pd.DataFrame(H)
            H['rule']=None
            for ii in H.index:
                if len(H.loc[ii,0])==1:
                    X=','.join(H.loc[ii,0])
                else:
                    X='('+','.join(H.loc[ii,0])+')'
                if len(H.loc[ii,1])==1:
                    Y=','.join(H.loc[ii,1])
                else:
                    Y='('+','.join(H.loc[ii,1])+')'
                rule='{}  --->>  {}'.format(X,Y)
                H.loc[ii,'rule']=rule
            H.rename(columns={0:'X',1:'Y',2:'sup',3:'conf',4:'lift'},inplace=True)
            H=pd.DataFrame(H,columns=['X','Y','rule','sup','conf','lift'])
            H['rank']=H['sup']*5+H['conf']
            H=H.sort_values(['sup','conf'],ascending=False)
            H=pd.DataFrame(H,columns=['X','Y','rule','sup','conf','lift'])
            H=H.reset_index(drop=True)
        else:
            H=None
        return (H,freqList)

    def difference(self, item, subset):
        return tuple(x for x in item if x not in subset)

    def confidence(self, subCount, itemCount):
        return float(itemCount)/subCount

    def support(self, count):
        return float(count)/self.numItems

    # 检查是否大于最小的支持度
    def firstPass(self, items, k):
        f = []
        for item, count in items.items():
            support = self.support(count)
            if support == 1:
                self.highSupportList.append(item)
            elif support >= self.minSup:
                f.append(item)

        return f

    def prepData(self):
        data=self.dataset
        data=data[data.T.notnull().all()]
        data=data.fillna(0)
        data=data.astype(bool)
        data=data[data.T.any()]
        data=data.loc[:,data.any()]
        columns=pd.Series(data.columns)
        k=list(data.index)
        v=[list(columns[list(data.loc[i,:])]) for i in k]
        self.numItems=len(data)
        self.transList=defaultdict(list,dict(zip(k,v)))
        self.itemset=set(data.columns)
        self.freqList=defaultdict(int,data.sum().to_dict())

