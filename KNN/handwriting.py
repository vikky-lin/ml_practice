import numpy as np 
from img2vec import *
from classify import *
from os import listdir

def handwriting():
    """
    desc:
        使用KNN算法实现手写数字识别
    总结:
        使用KNN算法不涉及训练数据，在apacheCN中有介绍，不细说。这里的文件是类图片形式，可以打开./input/中
        的一个文件看看。因为涉及到计算欧式距离，需要把每一个数字文件转化为一维的向量(感觉用numpy的多维数组计算也行，稍后试下)，
        于是就有了img2vec.py这个函数。
        完成第一部之后，classify.py中就是用一个未打标签的测试数据去计算与数据集的每一个数字的距离，取距离最近的k个数据，选取k
        个数据中次数最多的那个数字作为预测标签
        至于这个handwriting.py,只是classify.py的功能泛化,即通过classify.py的方法预测一批的未打标签的数据。
        其实这个说法是错误的，说是未打标签，实际上是也是有打了标签的，只不过我们要用这个标签去估计KNN算法模型的准确率。
    """
    path = './KNN/input/'
    train_file_list = listdir(path+'train_set/')
    train_vec_set = np.zeros((len(train_file_list),1024))
    train_vec_labels = []
    for i in range(len(train_file_list)):
        train_vec_set[i] = img2vec(path+'train_set/'+train_file_list[i])
        train_vec_labels.append(train_file_list[i].split('_')[0])

    test_file_list = listdir(path+'test_set/')
    test_vec_set = np.zeros((len(test_file_list),1024))
    test_vec_labels = []
    for i in range(len(test_file_list)):
        test_vec_set[i] = img2vec(path+'test_set/'+test_file_list[i])
        test_vec_labels.append(test_file_list[i].split('_')[0])
    
    error_count = 0
    for i in range(len(test_vec_set)):
        if classify(test_vec_set[i],train_vec_set,train_vec_labels,5) != test_vec_labels[i]:
            error_count += 1
            print('error locked at ',test_file_list[i])
    print('correct classifing rate is {}%'.format((len(test_vec_labels)-error_count)/len(test_vec_labels)*100))

handwriting()
