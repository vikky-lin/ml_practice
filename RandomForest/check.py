from buildRandomForest import buildRandomForest
from baggingPrediction import baggingPrediction
from crossValidationFolds import crossValidationFolds
from loadDataset import loadDataset
def evaluate_algorithm(dataset,k_folds):
    folds = crossValidationFolds(dataset,k_folds)
    for fold in folds:
        errorCount = 0
        trainSet = folds.copy()
        trainSet.remove(fold)
        trainSet = sum(trainSet,[])
        randomForest = buildRandomForest(trainSet,forestSize=15,n_features=15,maxDepth=10,minSize=1)
        for row in fold:
            # print(baggingPrediction(randomForest,row))
            if row[-1]!=baggingPrediction(randomForest,row):
                errorCount +=1
        print('errorRate is ',(errorCount/len(fold)))

if __name__ == "__main__":
    dataset = loadDataset('./RandomForest/dataset.txt')
    evaluate_algorithm(dataset,10)