from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import distanceDataset
import checkDatasets
import normalDataset
import pandas as pd
import classifier
import ageDataset
import os
import sys

DATA_POINTS = 25000
# TODO:
# - Fix results
#   - Decide on using graphing function that makes all graphs seperately for the three learning systems, or one that makes one graph for the three learning systems
#   - Comment code

def main():
    selectProgram()
    #runAll()

def selectParameters():
    for modelType in ["MLP"]:
        for testingSet in ["MFMF", "MFSF", "SFMF", "SFSF"]:
            if modelType == "MLP":
                for MLP in [2]:
                    df = pd.read_excel("DataRes/hyperparameters/" + modelType + testingSet + str(MLP) + ".xlsx")
                    for i in range(len(df.iter)):
                        if df.loc[i, 'rank_test_score'] == 1:
                            print(modelType + testingSet + str(MLP))
                            print(df.loc[i, 'mean_test_score'])
                            print(df.loc[i, 'params'])
            else:
                df = pd.read_excel("DataRes/hyperparameters/" + modelType + testingSet + "0.xlsx")
                for i in range(len(df.iter)):
                    if df.loc[i, 'rank_test_score'] == 1:
                        print(modelType + testingSet + "0")
                        print(df.loc[i, 'mean_test_score'])
                        print(df.loc[i, 'params'])
            print("==============================")

def runAll():
    learningSystems = ["MLP", "Random_Forest", "XGBoost"]
    datasets = ["Normal", "Age", "Contribution", "Spouse", "Residency", "Resource", "Distance"]
    allDatasets = []; out = []; models = []

    for i in range(0, len(learningSystems)):
        models.append(getModels(i))

    for j in range(0, len(datasets)):
        allDatasets.append(getDataset(j))

    allAcc = [[], [], []]
    for i in list(range(0, len(learningSystems))):
        for j in list(range(0, len(datasets))):
            out = runSetup(i, j, allDatasets[j], models[i])
            totalAcc = []
            for result in out:
                totalAcc.append(result[1])
            allAcc[i].append(totalAcc)

    # This is seperated because of printing issues 
    sys.stdout = open("DataRes/finalAccuracy.txt", "w")
    for i in list(range(0, len(learningSystems))):
        for j in list(range(0, len(datasets))):
            print(learningSystems[i] + datasets[j])
            print(allAcc[i][j])
            print("===========================================")

def getParameters():
    dat = normalDataset.getData(DATA_POINTS)
    learningSystems = ["MLP", "Random_Forest", "XGBoost"]
    
    for ls in list(range(0,3)):
        print("Training on multiple fail, testing on multiple fail"); classifier.findHyperParameters(dat[2], dat[3], learningSystems[ls], "MFMF")
        print("Training on multiple fail, testing on single fail"); classifier.findHyperParameters(dat[2], dat[1], learningSystems[ls], "MFSF")
        print("Training on single fail, testing on multiple fail"); classifier.findHyperParameters(dat[0], dat[3], learningSystems[ls], "SFMF")
        print("Training on single fail, testing on single fail"); classifier.findHyperParameters(dat[0], dat[1], learningSystems[ls], "SFSF")

def selectProgram():
    print("0 = MLP, 1 = Random Forest, 2 = XGBoost")
    ls = input("Which learning system?\n"); ls = int(ls)
    print("0 = Normal, 1 = Age, 2 = Contribution, 3 = Spouse, 4 = Residency, 5 = Resource, 6 = Distance")
    ds = input("Which dataset?\n"); ds = int(ds)
    if ls > 3 or 0 > ls or ds > 6 or 0 > ds:
        print("Error input value")
        exit()
    runSetup(ls, ds, getDataset(ds), getModels(ls))   

def getDataset(ds):
    dat = []
    if ds == 0:
        print("Getting normal dataset")
        dat = normalDataset.getData(DATA_POINTS)
    if ds == 1:
        print("Getting age dataset")
        dat = ageDataset.getData(DATA_POINTS)
    if ds > 1 and ds < 6:
        print("Getting one of the check datasets")
        dat = checkDatasets.getData(DATA_POINTS, ds)
    if ds == 6:
        print("Getting distance dataset")
        dat = distanceDataset.getData(DATA_POINTS)

    return dat

def getModels(ls):
    models = []
    if ls == 0:
        print("Getting MLP models")
        models = getMLPModels()
    if ls == 1:
        print("Getting Random Forest models")
        models = getRandomForestModels()
    if ls == 2:
        print("Getting XGBoost models")
        models = getXGBoostModels()

    return models

def runSetup(ls, ds, dat, models):
    learningSystems = ["MLP", "Random_Forest", "XGBoost"]
    datasets = ["Normal", "Age", "Contribution", "Spouse", "Residency", "Resource", "Distance"]
    name = str(str(datasets[ds]) + str(learningSystems[ls]))
    out = []
    if ds == 0:
        print("On normal dataset")
        out = runClassifierNormal(dat, models)
    if ds == 1:
        print("On age dataset")
        out = runClassifier(dat, models)
        ageDataset.printGraph(dat[2], out[0][0], 'SF' + name)
        ageDataset.printGraph(dat[2], out[1][0], 'MF' + name)
    if ds > 1 and ds < 6:
        print("One one of the check datasets")
        out = runClassifier(dat, models)
        #if ds > 1 and ds < 5:
            #checkDatasets.printBoolean(dat[2], out[0][0], 'SF' + name)
            #checkDatasets.printBoolean(dat[2], out[1][0], 'MF' + name)
        if ds == 5:
            checkDatasets.printNumericalGraph(dat[2], out[0][0], 'SF' + name)
            checkDatasets.printNumericalGraph(dat[2], out[1][0], 'MF' + name)
    if ds == 6:
        print("On distance dataset")
        out = runClassifier(dat, models)
        distanceDataset.printGraph(dat[2], out[0][0], 'SF' + name)
        distanceDataset.printGraph(dat[2], out[1][0], 'MF' + name)
    
    return out

# Function that creates the three MLP classifiers
def getMLPModels():
    models = []
    models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), activation = 'logistic', alpha = 0.008, learning_rate_init = 0.008, batch_size = 26, max_iter = 3000))
    #models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), activation = 'logistic', solver = 'adam', learning_rate_init = 0.001, batch_size = 50, max_iter = 50000))	

    return models

# Function that creates Random Forest classifiers
def getRandomForestModels():
    models = []
    models.append(RandomForestClassifier(n_estimators = 16, max_depth = 19, max_leaf_nodes = 17, min_samples_split = 6, random_state = 0))

    return models

# Function that creates XGBoost classifiers
def getXGBoostModels():
    models = []
    models.append(XGBClassifier(n_estimators = 16, max_depth = 7, objective ='reg:squarederror', learning_rate = 0.25, gamma = 0.5, verbosity = 0))

    return models

# Function that runs the normal dataset
def runClassifierNormal(datasets, models):
    print("Running the learning system on the normal dataset")
    print("Training on multiple fail, testing on multiple fail"); out1 = classifier.testDataset(datasets[2], datasets[3], models)
    print("Training on multiple fail, testing on single fail"); out2 = classifier.testDataset(datasets[2], datasets[1], models)
    print("Training on single fail, testing on multiple fail"); out3 = classifier.testDataset(datasets[0], datasets[3], models)
    print("Training on single fail, testing on single fail"); out4 = classifier.testDataset(datasets[0], datasets[1], models)

    return (out1, out2, out3, out4)

def runClassifier(datasets, models):
    print("Running the learning system")
    print("Training on single fail, testing on test dataset"); out1 = classifier.testDataset(datasets[0], datasets[2], models)
    print("Training on multiple fail, testing on test dataset"); out2 = classifier.testDataset(datasets[1], datasets[2], models)

    return (out1, out2)

if __name__ == "__main__":
    main()