from calEntropy import *
from findBestFeature import * 
from splitDataset import *
import numpy as np 

def createTree(dataset,featLabel):
    """
    desc:
        使用信息熵作为决策依据构建决策树
    """
    if len(np.unique(dataset[:,-1]))==1:
        return dataset[:,-1][0]
    bestFeatIdx = findBestFeature(dataset)
    bestFeat = featLabel[bestFeatIdx]
    tree = {bestFeat:{}}
    featLabel.pop(bestFeatIdx)
    uniqueLabel = set(dataset[:,bestFeatIdx])
    for label in uniqueLabel:
        tree[bestFeat][label] = createTree(splitDataset(dataset,bestFeatIdx,label),featLabel)
        print(tree)
    # if len(np.unique(dataset[:,-1]))>1:
    #     bestFeatIdx = findBestFeature(dataset)
    #     bestFeat = featLabel[bestFeatIdx]
    #     tree = {bestFeat:{}}
        # featLabel.pop(bestFeatIdx)
        # uniqueLabel = set(dataset[:,bestFeatIdx])
        # if len(featLabel)>0:
        #     for label in uniqueLabel:
        #         tree[bestFeat][label] = createTree(splitDataset(dataset,bestFeatIdx,label),featLabel)
        #         print(tree)
    #     return tree
    # return dataset[:,-1][0]
if __name__ == '__main__':
    with open('./DecisionTree/data/lenses.txt','r')as f:
        lines = f.readlines()
    dataset = np.array([line.strip('\n').split('\t') for line in lines])
    featLabel = ['age', 'prescript', 'astigmatic', 'tearRate']
    createTree(dataset,featLabel)