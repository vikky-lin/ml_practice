from buildTree import buildTree
from subSampling import subSampling
from getRandomFeature import getRandomFeature
from loadDataset import loadDataset
def buildRandomForest(dataset,forestSize=2,n_features=15,maxDepth=10,minSize=1,samplingSize=1):
    randomForest = []
    for index in range(forestSize):
        sample = subSampling(dataset,samplingSize)
        tree = buildTree(sample,n_features,maxDepth,minSize)
        randomForest.append(tree)
    return randomForest

if __name__ == "__main__":
    dataset = loadDataset('./RandomForest/dataset.txt')
    randomForest = buildRandomForest(dataset)
    print(randomForest)