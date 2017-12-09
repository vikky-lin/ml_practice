import numpy as np 
from img2vec import *
from classify import *
from os import listdir

def handwriting():
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
