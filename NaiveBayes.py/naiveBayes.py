import numpy as np 

def createWordLib():
    """
    desc:
        构建正常词汇与侮辱性词汇的单词列表，词汇量当然越大越好，
        这里按照正常词汇:侮辱性词汇=5:1的比例构建一个小的词库
        返回单词列表、词库以及单词标签
    注:
        为方便解释,下文所有的注释用正样本表示正常词汇，负样本表示侮辱性词汇
    """
    wordLib = [
        ['my','dog','has','flea','problems', 'help', 'please','it','good','day'
         'my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him',
         'mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him',
         'stop','posting','worthless''quit', 'buying', 'worthless','food',
          'maybe', 'not', 'take', 'him', 'to','park'],
        ['dog','stupid','fuck','nigar']
        ]
    wordCate = [0,1]
    # 初始化词库
    vocabLib = set()
    # 将单词列表中的单词添加到词库
    for item in wordLib:
        vocabLib = vocabLib | set(item)
    vocabLib = list(vocabLib)
    return wordLib,vocabLib,wordCate

def word2vec(vocabLib,input):
    """
    desc:
        将句子转成词向量
    """
    wordVec = np.zeros(len(vocabLib))
    for word in input:
        if word in vocabLib:
            wordVec[vocabLib.index(word)] = 1
    return wordVec

def train_naive_bayes(wordLib,vocabLib,wordCate,word2vec):
    """
    desc:
        训练词库，提取正样本和负样本的词向量，并计算负样本的先验概率
    """
    # 负样本的先验概率
    abPro = len(wordLib[1])/(len(wordLib[0])+len(wordLib[1]))
    # 初始化正负样本的词向量
    positiveVec = np.ones(len(vocabLib))
    negativeVec = np.ones(len(vocabLib))
    # 统计正负样本的单词数，apacheCN中修正版调整的positiveNum、negativeNum均为2，不是很明白原因，所以还是保持0
    positiveNum = 0
    negativeNum = 0
    for i in range(len(wordLib)):
        if wordCate[i] == 1:
            negativeVec += np.array(word2vec(vocabLib,wordLib[i]))
            negativeNum = len(wordLib[i])
        elif wordCate[i] == 0:
            positiveVec += np.array(word2vec(vocabLib,wordLib[i]))
            positiveNum = len(wordLib[i])
    # 归一化
    positiveVec = np.log(positiveVec/positiveNum)
    negativeVec = np.log(negativeVec/negativeNum)
    # print(negativeVec)
    return positiveVec,negativeVec,abPro

def calNaiveBayes(testVec,positiveVec,negativeVec,abPro):
    p = np.sum(testVec * positiveVec) + np.log(1-abPro)
    n = np.sum(testVec * negativeVec) + np.log(abPro)
    if p > n:
        print('没骂人')
    else:
        print('骂人了')

if __name__ == '__main__':
    test = 'you are stupid'
    wordLib,vocabLib,wordCate = createWordLib()
    testVec = word2vec(vocabLib,test.split())
    positiveVec,negativeVec,abPro = train_naive_bayes(wordLib,vocabLib,wordCate,word2vec)
    calNaiveBayes(testVec,positiveVec,negativeVec,abPro)