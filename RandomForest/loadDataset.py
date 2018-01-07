def loadDataset(path):
    """
    desc:
        加载数据集
    return:
        dataset:数据集
        classset:样本标签集
    """
    dataset = []
    classset = []
    with open(path,'r')as f:
        lines = f.readlines()
        for line in lines:
            dataset.append(line.strip().split(','))
            # classset.append(line.split(',')[-1])
    return dataset

if __name__ == "__main__":
    loadDataset('./RandomForest/dataset.txt')