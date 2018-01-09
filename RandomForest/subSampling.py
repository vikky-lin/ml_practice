from random import randrange
def subSampling(dataset,samplingSize):
    size = len(dataset)*samplingSize
    subDataset = []
    for i in range(size):
        index = randrange(0,size)
        subDataset.append(dataset[index])
    return subDataset