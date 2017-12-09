import numpy as np 
from math import log
def calEntropy(dataset):
    """
    desc:
        计算dataset的信息熵
    return:
        返回该dataset的entrop(香农熵)
    """
    labelList = []
    for featVec in dataset:
        labelList.append(featVec[-1])
    size = len(labelList) #数据集的记录数
    uniqueLabel = set(labelList)
    shannonEnt = 0 #初始化香农熵
    for label in uniqueLabel:
        prob = labelList.count(label)/size #计算该分类标签在整个数据集的出现频率
        shannonEnt -=prob*log(prob,2) #计算香农熵
    return shannonEnt

if __name__ == '__main__':
    dataset = np.array([[1,2,3],[2,1,2],[4,1,1],[2,2,1]])
    print(calEntropy(dataset))

