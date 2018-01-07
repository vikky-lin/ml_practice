from loadDataset import loadDataset
from getBestSplit import getBestSplit

def buildStump(dataset,depth=10,minSize=1):
    stump = {}
    root,groups = getBestSplit(dataset)
    currentDepth = 1
    while currentDepth<=10:
        stump{root:buildStump()}