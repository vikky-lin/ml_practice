import numpy as np 


def splitDataset(dataset,index,value):
    """
    desc:
        提取当前特征下某个取值(value)下的数据集(subDataset)
    dataset:
        数据集
    index:
        数据集的某个特征的标号
    value:
        该特征的某个取值(value)
    return:
        subDataset
    """
    # dataset的列数
    dim = dataset.shape[1]
    # dataset[:,index] == value 返回dataset的index列是否等于value的布尔矩阵
    # dataset[dataset[:,index] == value] 取dataset中布尔矩阵等于True的行
    # dataset[dataset[:,index] == value][:,[i for i in range(dim) if i != index]] 取dataset index列外的其他列
    subDataset = dataset[dataset[:,index] == value][:,[i for i in range(dim) if i != index]]
    return subDataset

if __name__ == '__main__':
    dataset = np.array([[1,2,3],[0,1,2],[1,2,3]])
    print(splitDataset(dataset,1,1))
