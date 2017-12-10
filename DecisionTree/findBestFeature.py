from calEntropy import *
from splitDataset import *
import numpy as np 

def findBestFeature(dataset):
    """
    desc:
        寻找dataset的最佳特征项
    """
    col = dataset.shape[1]
    row = dataset.shape[0]
    baseEnt = calEntropy(dataset)
    # print("当前dataset的信息熵：",baseEnt)
    infoGain = 0
    bestFeature = -1
    maxInfoGain = 0
    for i in range(col-1):
        uniqueLabel = set(dataset[:,i])
        labelCnt = {label:list(dataset[:,i]).count(label) for label in uniqueLabel}
        nowEnt = 0
        for label in uniqueLabel:
            subDataset = splitDataset(dataset,i,label)
            nowEnt += (labelCnt[label]/row)*calEntropy(subDataset)
        # print("第{}列的subDataset的信息熵：{}".format(i,nowEnt))
        infoGain = baseEnt - nowEnt
        # print("第{}列的信息增益：{}".format(i,infoGain))
        if infoGain > maxInfoGain:
            bestFeature = i
            maxInfoGain = infoGain
    return bestFeature

if __name__ == '__main__':
    with open('./DecisionTree/data/lenses.txt','r')as f:
        lines = f.readlines()
    dataset = np.array([line.strip('\n').split('\t') for line in lines])
    # print(findBestFeature(dataset))



