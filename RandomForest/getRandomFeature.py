import random
def getRandomFeature(sample,n_features):
    """
    desc:
        随机选取样本的n个特征变量，保证每个决策树都不相同的关键点之一
    """
    randomFeatures = []
    featureList = list(range(len(sample[0])-1))
    for i in range(n_features):
        index = random.randrange(len(featureList))
        randomFeatures.append(featureList[index])
        featureList.remove(featureList[index])          
    return randomFeatures

if __name__ == "__main__":
    print(getRandomFeature([[1,2,3,4,5],[1,2,3,4,5]],3))     
