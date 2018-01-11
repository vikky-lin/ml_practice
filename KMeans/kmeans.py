import numpy as np
def loadDatMat(path):   
    dataMat = []
    file = open(path)
    for line in file.readlines():
        curLine = line.strip().split()
        fltLine = list(map(float,curLine)) 
        dataMat.append(fltLine)
    return np.mat(dataMat)

def calDistance(vecA,vecB):
    """
    desc:
        计算两个向量间的欧式距离
    """
    
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

def randCentroid(dataMat,k):
    """
    desc:
        随机创建k个数据点作为数据集的起始质心，要求起始质心在数据集的边界之内。
        这种随机选择质心的方式可能产生较差的簇
    """
    dims = np.shape(dataMat)[1]  #矩阵的列数，也就是样本点的维度
    centroids = np.mat(np.zeros((k,dims)))   #创建k个起始质心矩阵
    for dim in range(dims):
        dimMean = np.mean(dataMat[:,dim])   #某一维度的最小值
        dimRange = np.float(np.max(dataMat[:,dim])-np.min(dataMat[:,dim]))/2    #某一维度的范围
        centroids[:,dim] = np.mat(dimMean+np.random.rand(k,1)*dimRange*np.random.randint(-1,1))   #随机生成k*1数据范围内的矩阵，并赋值给起始质心矩阵
    return centroids

def kmeans(dataMat,k):
    """
    desc:
        k-means 聚类算法
        该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
        这个过程重复数次，知道数据点的簇分配结果不再改变位置。
    """
    size = np.shape(dataMat)[0] #数据集大小，即矩阵行数
    centroids = randCentroid(dataMat,k) #起始质心矩阵
    clusterAssment = np.mat(np.zeros((size,2)))    # 创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果
    clusterChange = True    #簇改变标志
    while clusterChange:
        clusterChange = False
        for i in range(size):
            minDist = np.inf    #最小距离
            clusterIndex = -1 #数据点分配的簇索引
            for j in range(k):  
                distance = calDistance(centroids[j,:],dataMat[i,:])   #计算数据点到质心的欧式距离
                if distance<minDist:
                    minDist = distance
                    clusterIndex = j
            if clusterAssment[i,0] != clusterIndex:
                clusterChange = True
                clusterAssment[i,:] = clusterIndex,minDist**2   #保存分配的簇结果
        # print (centroids)
        #更新质心
        for cent in range(k):
            # print(clusterAssment[:,0]==cent)
            centData = dataMat[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            # print(np.nonzero(clusterAssment[:,0]==cent)[0])
            centroids[cent,:] = np.mean(centData,axis=0)
    return centroids,clusterAssment

def biskmeans(dataMat,k):
    #初始化一个质心
    centroid0 = np.mean(dataMat,axis=0).tolist()[0]
    #将数据点分配到该初始质心下，clusterAssment用来保存簇分配结果
    size = np.shape(dataMat)[0] #
    clusterAssment = np.mat(np.zeros((size,2)))
    # print(centroid0)
    #质心列表
    centroidsList = [centroid0]
    while len(centroidsList)<k:
        minSSE = np.inf
        for index in range(len(centroidsList)):
            #获取该簇的所有数据点集
            subDataMat = dataMat[np.nonzero(clusterAssment[:,0].A==index)[0]]
            #获取该簇分裂后的质心列表和簇分配结果
            subCentroids,subClusterAssment = kmeans(subDataMat,2)
            #计算该簇分裂后所有数据点的平方误差以及未参与分裂的其他簇的数据点的平方误差
            subSSE = np.sum(subClusterAssment[:,1])
            otherSSE = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=index)[0]][:,1])
            #计算总平方误差
            totalSSE = subSSE+otherSSE
            if totalSSE<minSSE:
                bestCentroidIndex = index   #保存最优的分裂质心下标
                bestCentroids = subCentroids   #保存该质心分裂后的两个质心位置
                bestClusterAssment = subClusterAssment  #保存该质心分裂后的数据点的簇分配结果
                minSSE = totalSSE
        #重新分配最优分裂质心的数据点的簇分配下标
        bestClusterAssment[np.nonzero(bestClusterAssment[:,0].A==1)[0],0]= len(centroidsList)
        bestClusterAssment[np.nonzero(bestClusterAssment[:,0].A==0)[0],0] = bestCentroidIndex
        #更新质心列表
        # centroidsList.remove(centroidsList[bestCentroidIndex])
        # centroidsList.extend(bestCentroids)
        centroidsList[bestCentroidIndex] = bestCentroids[0,:].tolist()[0]
        centroidsList.append(bestCentroids[1,:].tolist()[0])
        #更新数据点的簇分配结果
        clusterAssment[np.nonzero(clusterAssment[:,0].A==bestCentroidIndex)[0]] = bestClusterAssment
    return centroidsList,clusterAssment

if __name__ == "__main__":
    dataMat = loadDatMat('./KMeans/testSet.txt')
    #kmeans测试
    # print(dataMat)
    # centroids,clusterAssment = kmeans(dataMat,3)
    # print("centroids:",centroids)
    # print("clusterAssment:",clusterAssment)

    #biskmeans测试
    centroids,clusterAssment = biskmeans(dataMat,3)
    print("centroids:",centroids)
    print("clusterAssment:",clusterAssment)
