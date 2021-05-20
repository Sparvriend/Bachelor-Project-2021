from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import distanceDataset
import checkDatasets
import normalDataset
import classifier
import ageDataset
import sys
import time
import numpy as np
from datetime import timedelta

DATA_POINTS = 25000
LEARNING_SYSTEMS = ["MLP", "Random_Forest", "XGBoost"]
DATASET_NAMES = ["Normal", "Age", "Contribution", "Spouse", "Residency", "Resource", "Distance"]

# TODO:
# - Comment code
# - Make graphs so that it sums over iterations
# - Combine special datasets into one .py file?

def main():
    #selectProgram()
    #runAll()
    multipleRuns(10)

def testModels(trainedModels, allDatasets, scalers):
    allAcc = [[], [], []]
    for i in list(range(len(LEARNING_SYSTEMS))):
        totalAcc = []
        # The normal computation is done manually
        out = [classifier.onlyTest(trainedModels[i][0], allDatasets[0][1], scalers[1]), classifier.onlyTest(trainedModels[i][0], allDatasets[0][0], scalers[0]),
               classifier.onlyTest(trainedModels[i][1], allDatasets[0][1], scalers[1]), classifier.onlyTest(trainedModels[i][1], allDatasets[0][0], scalers[0])]
        
        for result in out:
            totalAcc.append([result[1]])
        allAcc[i].append(totalAcc)

        for j in list(range(len(DATASET_NAMES)))[1:]:
            out = [classifier.onlyTest(trainedModels[i][0], allDatasets[j], scalers[0]), classifier.onlyTest(trainedModels[i][1], allDatasets[j], scalers[1])]
            totalAcc = []
            for result in out:
                totalAcc.append([result[1]])
            allAcc[i].append(totalAcc)

    return allAcc

def printFinalAccuracy(allAcc):
    # This is seperated from the other loops because of printing issues 
        sys.stdout = open("DataRes/finalAccuracy.txt", "w")
        for i in list(range(len(LEARNING_SYSTEMS))):
            for j in list(range(len(DATASET_NAMES))):
                print(LEARNING_SYSTEMS[i] + DATASET_NAMES[j]); print(allAcc[i][j])
                print("===========================================")

def runAll(PRINT_TO_FILE = 1):
    allDatasets = []; trainedModels = []

    train = normalDataset.getOnlyTrain(DATA_POINTS)
    scalers = getScalers(train[0], train[1])

    for i in range(len(LEARNING_SYSTEMS)):
        trained = []; models = getModels(i)
        trained.append(classifier.trainModel(train[0], models[0], scalers[0]))
        trained.append(classifier.trainModel(train[1], models[1], scalers[1]))
        trainedModels.append(trained)

    for j in range(len(DATASET_NAMES)):
        allDatasets.append(getOnlyTest(j))

    allAcc = testModels(trainedModels, allDatasets, scalers)

    if PRINT_TO_FILE == 1:
        printFinalAccuracy(allAcc)

    return allAcc

def multipleRuns(iterations):
    startTime = time.time()
    allResults = []; totalStDev = []; wrongAccBool = False

    for p in range(iterations):
        print("################")
        print("ON ITERATION: " + str(p))
        print("################")
        res = runAll(0)
        if p == 0:
            allResults = res
            if res[0][0][0][0] < 0.6:
                wrongAccBool = True
        else:
            if res[0][0][0][0] < 0.6:
                wrongAccBool = True
            for i in list(range(len(LEARNING_SYSTEMS))):
                for j in list(range(len(DATASET_NAMES))):
                    for q in list(range(len(res[i][j]))):
                        allResults[i][j][q].append(res[i][j][q])

    for i in list(range(len(LEARNING_SYSTEMS))):
        totalStDev.append([])
        for j in list(range(len(DATASET_NAMES))):
            totalStDev[i].append([])
            for q in list(range(0, len(res[i][j]))):
                totalStDev[i][j].append([])
                totalStDev[i][j][q] = round(float(np.std(allResults[i][j][q])), 2)
                allResults[i][j][q] = round(float(np.mean(allResults[i][j][q])), 2)
                
    print(wrongAccBool)
    print("Time elapsed: " + str(timedelta(seconds = time.time() - startTime)))
    sys.stdout = open("DataRes/totalAccuracy" + str(iterations) + "iterations.txt", "w")

    for i in list(range(len(LEARNING_SYSTEMS))):
        for j in list(range(len(DATASET_NAMES))):
            print(LEARNING_SYSTEMS[i] + DATASET_NAMES[j])
            print(allResults[i][j])
            print(totalStDev[i][j])
            print("===========================================")

def getScalers(singleFail, multipleFail):
    scalers = []
    x_train = singleFail.drop(['Age', 'Resource', 'Distance', 'Eligible'], axis = 1)
    scaler = MinMaxScaler(); scaler.fit(x_train)
    scalers.append(scaler)

    x_train = multipleFail.drop(['Age', 'Resource', 'Distance', 'Eligible'], axis = 1)
    scaler = MinMaxScaler(); scaler.fit(x_train)
    scalers.append(scaler)
    return scalers

def selectProgram():
    print("0 = MLP, 1 = Random Forest, 2 = XGBoost")
    ls = input("Which learning system?\n"); ls = int(ls)
    print("0 = Normal, 1 = Age, 2 = Contribution, 3 = Spouse, 4 = Residency, 5 = Resource, 6 = Distance")
    ds = input("Which dataset?\n"); ds = int(ds)
    if ls > 3 or 0 > ls or ds > 6 or 0 > ds:
        print("Error input value")
        exit()
    runSingle(ls, ds, normalDataset.getOnlyTrain(DATA_POINTS), getOnlyTest(ds), getModels(ls))   

def getOnlyTest(ds):
    if ds == 0:
        print("Getting normal test dataset")
        return normalDataset.getOnlyTest(DATA_POINTS)
    if ds == 1:
        print("Getting age test dataset")
        return ageDataset.getOnlyTest(DATA_POINTS)
    if ds > 1 and ds < 6:
        print("Getting one of test check datasets")
        return checkDatasets.getOnlyTest(DATA_POINTS, ds)
    if ds == 6:
        print("Getting test distance dataset")
        return distanceDataset.getOnlyTest(DATA_POINTS)

def getModels(ls):
    models = []
    if ls == 0:
        print("Getting MLP models")
        models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), activation = 'logistic', alpha = 0.00008, learning_rate_init = 0.008, batch_size = 26, max_iter = 3000))
        models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), activation = 'logistic', alpha = 0.00008, learning_rate_init = 0.008, batch_size = 26, max_iter = 3000))
        # Cor's models:
        # models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), activation = 'logistic', solver = 'adam', learning_rate_init = 0.001, batch_size = 50, max_iter = 50000))
        # models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), activation = 'logistic', solver = 'adam', learning_rate_init = 0.001, batch_size = 50, max_iter = 50000))
    if ls == 1:
        print("Getting Random Forest models")
        models.append(RandomForestClassifier(n_estimators = 16, max_depth = 19, max_leaf_nodes = 17, min_samples_split = 6, random_state = 0))
        models.append(RandomForestClassifier(n_estimators = 16, max_depth = 19, max_leaf_nodes = 17, min_samples_split = 6, random_state = 0))
    if ls == 2:
        print("Getting XGBoost models")
        models.append(XGBClassifier(n_estimators = 16, max_depth = 7, objective ='reg:squarederror', learning_rate = 0.25, gamma = 0.5, verbosity = 0))
        models.append(XGBClassifier(n_estimators = 16, max_depth = 7, objective ='reg:squarederror', learning_rate = 0.25, gamma = 0.5, verbosity = 0))

    return models

def runSingle(ls, ds, train, test, models):
    name = str(str(DATASET_NAMES[ds]) + str(LEARNING_SYSTEMS[ls])); out = []; trainedModels = []
    scalers = getScalers(train[0], train[1])
    trainedModels = [classifier.trainModel(train[0], models[0], scalers[0]), classifier.trainModel(train[1], models[1], scalers[1])]  
    if ds == 0:
        print("On normal dataset")
        out = runClassifierNormal(trainedModels, test, scalers)
    if ds == 1:
        print("On age dataset")
        out = runClassifier(trainedModels, test, scalers)
        ageDataset.printGraph(test, out[0][0], 'SF' + name)
        ageDataset.printGraph(test, out[1][0], 'MF' + name)
    if ds > 1 and ds < 6:
        print("One one of the check datasets")
        out = runClassifier(trainedModels, test, scalers)
        if ds == 5:
            checkDatasets.printNumericalGraph(test, out[0][0], 'SF' + name)
            checkDatasets.printNumericalGraph(test, out[1][0], 'MF' + name)
    if ds == 6:
        print("On distance dataset")
        out = runClassifier(trainedModels, test, scalers)
        distanceDataset.printGraph(test, out[0][0], 'SF' + name)
        distanceDataset.printGraph(test, out[1][0], 'MF' + name)

# Function that runs the normal dataset
def runClassifierNormal(trainedModels, testSets, scalers):
    print("Training on multiple fail, testing on multiple fail"); out1 = classifier.onlyTest(trainedModels[0], testSets[1], scalers[1])
    print("Training on multiple fail, testing on single fail"); out2 = classifier.onlyTest(trainedModels[0], testSets[0], scalers[0])
    print("Training on single fail, testing on multiple fail"); out3 = classifier.onlyTest(trainedModels[1], testSets[1], scalers[1])
    print("Training on single fail, testing on single fail"); out4 = classifier.onlyTest(trainedModels[1], testSets[0], scalers[0])

    return (out1, out2, out3, out4)

def runClassifier(trainedModels, testSet, scalers):
    print("Running single fail system on test dataset"); out1 = classifier.onlyTest(trainedModels[0], testSet, scalers[0])
    print("Running multiple fail system on test dataset"); out2 = classifier.onlyTest(trainedModels[1], testSet, scalers[1])

    return (out1, out2)

if __name__ == "__main__":
    main()