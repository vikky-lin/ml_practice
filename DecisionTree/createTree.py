from calEntropy import *
from findBestFeature import * 
from splitDataset import *
import numpy as np 

def createTree(dataset,featLabel):
    """
    desc:
        使用信息熵作为决策依据构建决策树
    总结:
        使用信息熵作为决策依据的决策树算法建立步骤(暂不考虑过度拟合等问题)
        1.预处理数据集，从文本文件中读取数据并存储为numpy中的数组结构,因为这里的文件中第一行不是列标签，
          所以需要新建一个list存储每一列的标签
        2.建立决策树是由各个环节组合而成的，比如计算该列的信息熵，切分数据集等；先完成每一个小的环节，
          最后通过createTree这个函数去调用每一个环节，下面列一下三个环节
          1)计算信息熵(见calEntropy.py)
          2)寻找最优列，这里会用到1)中的信息熵计算函数
          3)找到最优列后，就是根据这一列的取值切分数据集，举例：
            性别  职业  年龄  信用度
            男   白领  28  高
            男   工人  58  低
            女   白领  35  高
            女   家庭主妇    48 中
            如果最优列是'性别'，那么根据这列的取值可以将数据集切分成
            '男':
                职业  年龄  信用度
                白领  28  高
                工人  58  低
            '女':
                职业  年龄  信用度
                白领  35  高
                家庭主妇    48 中
            这两个子数据集
        3.完成上面的三个环节的函数定义后，就可以通过这里的createTree来建立最终的决策树了
    """
    # 设置剪枝条件
    # 这里设置的条件是最简单的一种情况，即当前数据集的分类标签只有一种时则返回此分类标签
    # 可以设置一个阈值，如分类标签中其中一种标签的占比达到80%以上时就返回这种标签
    if len(np.unique(dataset[:,-1]))==1:
        return dataset[:,-1][0]
    # 寻找最优列，从featLabel中获取最优列的标签名
    bestFeatIdx = findBestFeature(dataset)
    bestFeat = featLabel[bestFeatIdx]
    # 初始化该决策树
    tree = {bestFeat:{}}
    # 从featLabel删除该标签
    del featLabel[bestFeatIdx]
    # 获取最优列去重后的value集合，并对此value集合递归调用createTree(dataset,featLabel)函数，
    # 这里的dataset是该value下的数据集，所以要用splitDataset(dataset,index,value)去获取子数据集
    uniqueValue = set(dataset[:,bestFeatIdx])
    for value in uniqueValue:
        subLabel = featLabel[:]
        tree[bestFeat][value] = createTree(splitDataset(dataset,bestFeatIdx,value),subLabel)
    # 打印最终的决策树
    print(tree)
    return tree
if __name__ == '__main__':
    with open('./DecisionTree/data/lenses.txt','r',encoding='utf-8')as f:
        lines = f.readlines()
    dataset = np.array([line.strip('\n').split('\t') for line in lines])
    featLabel = ['age', 'prescript', 'hyper', 'tearRate']
    createTree(dataset,featLabel)