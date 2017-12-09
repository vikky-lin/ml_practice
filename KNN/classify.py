import numpy as np
import operator

def classify(test_vec,train_vec_set,train_vec_labels=None,k=3):
    """
    desc:
        KNN算法：计算测试样本与训练集各个样本的欧式距离,选取距离最小的k个训练样本并对选中样本的标签计数,
        最多的标签作为测试样本的预测标签输出
    test_vec:
        测试样本的向量
    train_vec_set:
        训练样本向量集
    train_vec_labels:
        训练样本标签
    k:
        选取的参数k    
    """
    distance = (((np.tile(test_vec,(len(train_vec_set),1)) - train_vec_set)**2).sum(axis=1))**0.5
    sorted_distance = np.argsort(distance)
    count = {}
    for i in range(k):
        NN = train_vec_labels[sorted_distance[i]]
        count[NN] = count.get(NN,0)+1
    return sorted(count.items(),reverse=True,key=operator.itemgetter(1))[0][0]

if __name__ == '__main__':
    test_vec = np.array([1,2,3])
    train_vec_set = np.array([[1,2,2],[5,3,7],[2,3,1],[4,6,2]])
    train_vec_labels = ['i','am','lin','jun']
    result = classify(test_vec,train_vec_set,train_vec_labels,3)
    print("预测输出标签为：",result)