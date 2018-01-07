def leaf(dataset):
    """
    desc:
        返回数据集中最多的类标签作为叶节点
    """
    classLabels = [row[-1] for row in dataset]
    return max(set(classLabels),key=classLabels.count)
