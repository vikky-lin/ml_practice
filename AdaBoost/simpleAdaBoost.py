import numpy as np 

def getWeakClassifier(dataMat,classMat,weightMat):
    """
    desc:
        在当前样本权重下，选择最佳的分类特征(如果样本有多个特征)和阈值，使得分类的错误率最低。得到一个弱分类器模型。
        这里的错误率不仅和错误样本数有关，还和错误样本的权重有关。当权重较大的样本分类正确时，
        即使有好几个权重小的样本都分类错了，最后的错误率也可能是最低的。
        比如，高考时候计算理综的成绩，假设总分100分，物理、化学、生物的比重分别为：0.6、0.2、0.2。
        那么想要获得一个好的成绩，物理成绩就特别重要。如果物理成绩低，那么即使化学、生物成绩再好，
        总成绩也不会好到哪去。
    dataMat:输入数据矩阵
    classMat:分类标签矩阵
    weights:样本权重矩阵
    return:
        weakClassifier:{dim:特征列,thredhold:阈值}
        errorRate:该分类器的错误率
        classPre:该分类器的样本预测分类
    """
    # 初始化弱分类器
    weakClassifier = {}
    #获取特征列数目
    dim = np.shape(dataMat)[1]
    # print(dim)
    #遍历各特征列
    for i in range(dim):
        #获取当前特征列的最大值与最小值
        rangeMin = dataMat[:,i].min()
        rangeMax = dataMat[:,i].max()
        #设置阈值增量
        numSteps = 20
        stepSize = (1+rangeMax-rangeMin)/numSteps
        # stepSize = 0.5
        # print(stepSize)
        #初始化最小误差为无穷大
        minError = np.inf
        for j in range(-1,numSteps+1):
            thredhold = rangeMin+j*stepSize
            # print('阈值：',thredhold)
            # print(thredhold)
            classPre,errorRate,inequal = check(dataMat,classMat,i,weightMat,thredhold)
            # print('错误率：',classPre.T)
            if  errorRate < minError:
                minError = errorRate
                classEst = classPre
                # print('预测值',classPre.T)
                weakClassifier['bestFeat'] = i
                weakClassifier['thredhold'] = thredhold
                weakClassifier['inequal'] = inequal
    print('此轮的弱分类器：',weakClassifier)
    # print(minError)
    alpha = 0.5*np.log((1-minError)/max(minError, 1e-16))   #等式中的1e-16主要用来避免出现minError为0时分母为0的情况
    print('alpha:',alpha)
    weakClassifier['alpha'] = alpha
    print('该轮预测值',classEst.T)
    Z= (weightMat.T)*(np.exp(-1*alpha*(np.multiply(classMat,classEst))))
    # Z = 1
    # print('Z:',Z.T)
    #权重更新等式中的分子部分
    expon = np.multiply(-1*alpha*classMat,classEst)
    # print('expon:',expon.T)
    W = np.multiply(weightMat,np.exp(expon))
    # print('W:',W.T)
    weightMat = W/Z
    print('更新权重:',weightMat.T)
    return weightMat,weakClassifier

def check(dataMat,classMat,dim,weightMat,thredhold):
    """
    desc:
        在给定样本权重前提下，计算某特征列在某一个阈值的分类情况以及分类错误率
    return:
        classPre:分类结果
        errorRate:分类错误率
        inequal:负类取向
    """
    #样本数
    m = np.shape(dataMat)[0]
    # print(m)
    #≥阈值为负类时的预测标签矩阵及错误率
    classPre_lt = np.mat(np.ones((m,1)))    #将样本的分类默认都设置为1，下同
    classPre_lt[dataMat[:,dim]>=thredhold] = -1
    errorMat_lt = np.mat(np.ones((m,1)))
    errorMat_lt[classPre_lt==classMat] = 0  
    errorRate_lt = (errorMat_lt.T)*weightMat    #根据各样本权重计算最终的分类错误率，下同
    # print('大于阈值:',errorRate_lt)
    #<阈值为负类时的预测标签矩阵及错误率
    classPre_gt = np.mat(np.ones((m,1)))
    classPre_gt[dataMat[:,dim]<thredhold] = -1
    errorMat_gt = np.mat(np.ones((m,1)))
    errorMat_gt[classPre_gt==classMat] = 0
    errorRate_gt = (errorMat_gt.T)*weightMat
    # 取分类错误率较低者作为样本在该阈值的分类结果
    if errorRate_lt[0,0] < errorRate_gt[0,0]:
        classPre = classPre_lt
        errorRate = errorRate_lt[0,0]
        inequal = 'lt'
    else:
        classPre = classPre_gt
        errorRate = errorRate_gt[0,0]
        inequal = 'gt'
    return classPre,errorRate,inequal

def trainDataset(dataMat,classMat,weightMat):
    # 弱分类器集合
    weakClassifierSet = []
    m = np.shape(dataMat)[0]
    while 1:
        errorMat = np.mat(np.ones((m,1)))
        W,weakClassifier = getWeakClassifier(dataMat,classMat,weightMat)
        weakClassifierSet.append(weakClassifier)
        classPre = adaBoost(dataMat,classMat,weakClassifierSet)
        errorMat[classPre==classMat] = 0  
        if errorMat.sum()==0:
            break
        else:
            weightMat = W
        print('**************************')
    return weakClassifierSet

def adaBoost(dataMat,classMat,weakClassifierSet):
    #样本数
    m = np.shape(dataMat)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for weakClassifier in weakClassifierSet:     
        classPre = np.mat(np.ones((m,1)))
        if weakClassifier['inequal'] == 'lt':
            classPre[dataMat[:,weakClassifier['bestFeat']] >= weakClassifier['thredhold']] = -1
        else:
            classPre[dataMat[:,weakClassifier['bestFeat']] < weakClassifier['thredhold']] = -1
        aggClassEst += classPre*weakClassifier['alpha']
    return np.sign(aggClassEst)
def test(weakClassifierSet,testData):
    """
    desc:
        用训练好的弱分类器集合检测新样本，返回预测分类结果
    """
    #score为样本的分类得分，大于等于0分类为1，反之为-1
    score = 0
    for weakClassifier in weakClassifierSet:
        if weakClassifier['inequal'] == 'lt':
            if testData[weakClassifier['bestFeat']]>= weakClassifier['thredhold']:
                score += -1*weakClassifier['alpha']
            else:
                score += 1*weakClassifier['alpha']
        else:
            if testData[weakClassifier['bestFeat']]< weakClassifier['thredhold']:
                score += -1*weakClassifier['alpha']
            else:
                score += 1*weakClassifier['alpha']
    if score >= 0:
        print('样本分类为：',1)
    else:
        print('样本分类为：',-1)
    return np.sign(score)

if __name__ == "__main__":
    dataMat = np.mat(np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.])).T
    classMat = np.mat(np.array([1.,1.,1.,-1.,-1.,-1.,1.,1.,1.,-1.])).T
    #初始化样本权重
    m = np.shape(dataMat)[0]
    weightMat = np.mat(np.ones(m)/m).T
    weakClassifierSet = trainDataset(dataMat,classMat,weightMat)
    testData = np.mat(np.array([-1])).T
    test(weakClassifierSet,testData)
