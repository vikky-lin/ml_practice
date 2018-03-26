import numpy as np 

def loadDataset(path):
    with open(path,'r')as f:
        data = f.readlines()
    Dataset = []
    classLabel = []
    for line in data:
        sample = line.strip().split('\t')
        Dataset.append([np.float(1),np.float(sample[0]),np.float(sample[1])])
        classLabel.append([sample[-1]])
    dataMat = np.mat(Dataset)
    labelMat = np.mat(classLabel)
    # print(np.shape(labelMat))
    return dataMat,labelMat

def sigmoid(x):
    return 1/(1+np.exp(-x))

def plot_best_fit(dataMat,labelMat,weights):
    """
    可视化
    :param weights: 
    :return: 
    """
    import matplotlib.pyplot as plt
    data_mat, label_mat = dataMat,labelMat
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if label_mat[i] == '1':
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
    weights = np.array(weights.transpose())
    print(weights)
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    """
    y的由来，卧槽，是不是没看懂？
    首先理论上是这个样子的。
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

def grdAscent(alpha=0.01,W=[0,0,0],maxCycle=400):
    W = np.mat(W)
    dataMat,labelMat = loadDataset('./Logistic/input/TestSet.txt')
    # print(np.shape(dataMat))
    for i in range(maxCycle):
        prob = sigmoid(W*(dataMat.transpose()))
        error = labelMat.astype('float').transpose()-prob
        W = W + alpha*error*dataMat
    # print(np.shape(dataMat[0]))
    
    plot_best_fit(dataMat,labelMat,W)
    errorCount = 0
    for i in range(100):
        prediction = sigmoid(W*(dataMat[i].transpose()))
        if prediction>=0.5:
            label = '1'
            if label != labelMat[i]:
                errorCount+=1
        else:
            label = '0'
            if label != labelMat[i]:
                errorCount+=1
        print("样本{}预测分类为{},实际分类为{}".format(i+1,label,labelMat[i]))
    print(errorCount)
    
    return W
if __name__ == '__main__':
    # loadDataset('./Logistic/input/TestSet.txt')
    grdAscent()