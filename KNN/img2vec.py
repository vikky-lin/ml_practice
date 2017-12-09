from numpy import *

def img2vec(file):
    """
    desc:
        将储存在txt中的数字转换成一维向量
    file:
        手写数字的文件
    return:
        返回该数字的一维向量
    """
    with open(file,'r')as f:
        vec = []
        lines = f.readlines()
        for line in lines:
            vec.append([int(i) for i in line.strip()])
        vec = reshape(vec,(1,len(lines[0].strip())*len(lines)))[0]
    return vec

if __name__ == '__main__':
    print(img2vec('./KNN/0_0.txt'))

