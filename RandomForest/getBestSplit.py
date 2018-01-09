import math
from loadDataset import loadDataset
from getRandomFeature import getRandomFeature
def getBestSplit(dataset,n_features):
    """
    desc:
        获取当前数据集的最佳特征极其最佳分割点
    """
    classLabels = set(sample[-1] for sample in dataset)  #当前数据集的类标签
    bestScore = math.inf
    bestFeature = None
    thredhold = math.inf
    randomFeatureList = getRandomFeature(dataset,n_features)
    for index in randomFeatureList:  #遍历样本所有特征
        for row in dataset:
            left = []
            right = []
            thredhold = row[index]  #当前阈值            
            gini = 0    #根据此阈值划分数据集的基尼系数
            for row in dataset:     #根据阈值将数据集划分为两部分 
                if row[index]<thredhold:
                    left.append(row)
                else:
                    right.append(row)
            classLabel_left = [row[-1] for row in left] #左子节点类标签
            classLabel_right = [row[-1] for row in right]   #右子节点类标签
            for classLabel in classLabels:  #计算当前特征阈值下的gini系数
                if len(left) == 0 and len(right) == 0:
                    continue
                else:
                    proportion_left = classLabel_left.count(classLabel)/max(1,len(left))
                    proportion_right = classLabel_right.count(classLabel)/max(1,len(right))
                    gini += proportion_left*(1-proportion_left) + proportion_right*(1-proportion_right)
            if gini < bestScore:
                bestScore = gini
                bestFeature = index
                thredhold = thredhold
    bestSplit = {"bestFeature":bestFeature,"thredhold":thredhold,"groups":(left,right),"size":len(dataset)}
    return bestSplit
if __name__ == "__main__":
    dataset = loadDataset('./RandomForest/dataset.txt')
    b,g = getBestSplit(dataset,1)
    print(b)