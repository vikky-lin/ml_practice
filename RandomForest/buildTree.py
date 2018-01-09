from loadDataset import loadDataset
from getBestSplit import getBestSplit
from splitNode import splitNode
def buildTree(dataset,n_features,maxDepth,minSize):
    """
    desc:
        构建单个决策树
    """
    root= getBestSplit(dataset,n_features)
    splitNode(root,n_features,maxDepth,minSize,1)
    return root


if __name__ == "__main__":
    dataset = loadDataset('./RandomForest/dataset.txt')
    print(buildTree(dataset,n_features=10,maxDepth=10,minSize=1))