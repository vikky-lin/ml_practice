import random
def getRandomFeature(sample,n_features):
    """
    desc:
        随机选取样本的n个特征变量，保证每个决策树都不相同的关键点之一
    """
    randomFeatures = []
    for i in range(n_features):
        while 1:
            index = random.randrange(len(sample)-1)
            if index not in randomFeatures:
                randomFeatures.append(index)
                break            
    return randomFeatures

if __name__ == "__main__":
    print(getRandomFeature([1,2,3,4,5],3))          
