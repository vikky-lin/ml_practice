from buildRandomForest import buildRandomForest
from predict import predict
def baggingPrediction(randomForest,row):
    """
    desc:
        用训练好的随机森林模型使用装袋技术对输入数据的分类进行预测，少数服从多数
    randomForest:
        随机森林模型
    row:
        输入数据
    return:
        预测分类结果
    """
    prediction = []
    for tree in randomForest:
        prediction.append(predict(tree,row))
    calssSet = set(prediction)
    return max(calssSet,key=prediction.count)
                 