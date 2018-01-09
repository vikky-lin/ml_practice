import random
def crossValidationFolds(dataset,k_folds):
    """
    desc:
        将数据集有放回的抽样，得到k份样本
    """
    dataset_copy = dataset.copy()
    #每份样本的样本大小
    foldSize = int(len(dataset)/k_folds)
    #样本
    folds = []
    for i in range(k_folds):
        fold = []
        for j in range(foldSize):
            index = random.randrange(0,len(dataset_copy))
            fold.append(dataset_copy[index])
        folds.append(fold)
    return folds


