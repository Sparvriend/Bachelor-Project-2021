from joblib import Parallel, delayed
from datetime import timedelta
import distanceDataset
import checkDatasets
import normalDataset
import numpy as np
import classifier
import ageDataset
import sys
import time

# This file is used as a setup point for the entire project. Many different things are done in this one general file.
# The multipleRuns function runs an amount of ITERATIONS, which then produces accuracy, standard deviation and graphs as output.

# Project constants. Note that datasets are DATA_POINTS*2 in size, this is due to most datasets being 50% eligible, hence it being only half.
# LEARNING_SYSTEMS and DATASET_NAMES are only used for naming graphs and for getting the amount of datasets/learning systems to test on.
DATA_POINTS = 25000
ITERATIONS = 100
LEARNING_SYSTEMS = ["MLP", "Random_Forest", "XGBoost"]
DATASET_NAMES = ["Normal", "Age", "Contribution", "Spouse", "Residency", "Resource", "Distance"]

def main():
    multipleRuns()

# This function runs multiple iterations of runAll(), namely in runIterationsPar().
# It runs everything and then finalizes everything by calculating standard deviations, accuracies and the graphs.
def multipleRuns():
    startTime = time.time(); totalStDev = []
    summedResultArrs, allResults = runIterationsPar()

    for i in range(len(LEARNING_SYSTEMS)):
        totalStDev.append([])
        for j in range(len(DATASET_NAMES)):
            totalStDev[i].append([])
            for q in range(0, len(allResults[i][j])):
                totalStDev[i][j].append([])
                totalStDev[i][j][q] = round(float(np.std(allResults[i][j][q]))*100, 4)
                allResults[i][j][q] = round(float(np.mean(allResults[i][j][q]))*100, 4)

    printGraphs(divAll(summedResultArrs))
    print("Time elapsed: " + str(timedelta(seconds = time.time() - startTime)))
    sys.stdout = open("DataRes/totalAccuracy" + str(ITERATIONS) + "iterations.txt", "w")

    for i in range(len(LEARNING_SYSTEMS)):
        for j in range(len(DATASET_NAMES)):
            print(LEARNING_SYSTEMS[i] + DATASET_NAMES[j])
            print(allResults[i][j]); print(totalStDev[i][j])
            print("===========================================")

# Function that runs ITERATIONS in parallel by using the Parallel package.
# It combines the results of each iteration.
def runIterationsPar():
    input = Parallel(n_jobs=-8)(delayed(runIteration)() for i in range(ITERATIONS))
    total = input[0][0]; outResults = input[0][1]

    for i in range(len(input))[1:]:
        total = addToSum(total, input[i][0])
        for j in range(len(LEARNING_SYSTEMS)):
            for p in range(len(DATASET_NAMES)):
                for q in range(len(input[i][1][j][p])):
                    outResults[j][p][q].append(input[i][1][j][p][q])
    return total, outResults

def runIteration():
    res = runAll()
    return res[0], res[1]

# Function that runs every learning system on every dataset. The specifics of running everything is defined in testModels.
def runAll():
    train = normalDataset.getOnlyTrain(DATA_POINTS); scalers = classifier.getScalers(train[0], train[1])
    trainedModels = trainModels(train, scalers); allDatasets = []

    for j in range(len(DATASET_NAMES)):
        allDatasets.append(getOnlyTest(j))
    res = testModels(trainedModels, allDatasets, scalers)

    return res

# Function that defines the training of the machine learning models.
def trainModels(train, scalers):
    trainedModels = []
    for i in range(len(LEARNING_SYSTEMS)):
        trained = []; models = classifier.getModels(i)
        trained.append(classifier.trainModel(train[0], models[0], scalers[0]))
        trained.append(classifier.trainModel(train[1], models[1], scalers[1]))
        trainedModels.append(trained)
    return trainedModels

# Function that retrieves testing datasets based on the input value.
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

# Function that test all the datasets on two trained models.
# It adds up all the different results and returns the accuracies and predictions of the models.
def testModels(trainedModels, allDatasets, scalers):
    allAcc = [[], [], []]; allResultArrs = []; fullResultArrs = []
    for i in range(len(LEARNING_SYSTEMS)):
        totalAcc = []
        # The normal computation is done manually
        out = [classifier.onlyTest(trainedModels[i][0], allDatasets[0][1], scalers[1]), classifier.onlyTest(trainedModels[i][0], allDatasets[0][0], scalers[0]),
               classifier.onlyTest(trainedModels[i][1], allDatasets[0][1], scalers[1]), classifier.onlyTest(trainedModels[i][1], allDatasets[0][0], scalers[0])]
        
        for result in out:
            totalAcc.append([result[1]])
        allAcc[i].append(totalAcc)

        for j in range(len(DATASET_NAMES))[1:]:
            out = [classifier.onlyTest(trainedModels[i][0], allDatasets[j], scalers[0]), classifier.onlyTest(trainedModels[i][1], allDatasets[j], scalers[1])]
            totalAcc = []
            
            for result in out:
                totalAcc.append([result[1]])
            allAcc[i].append(totalAcc)

            totalResultArr = [getResultArr(out[0][0], allDatasets[j], j), getResultArr(out[1][0], allDatasets[j], j)]
            if totalResultArr != [None, None]:         
                allResultArrs.append(totalResultArr)
        fullResultArrs.append(allResultArrs); allResultArrs = []
    return (fullResultArrs, allAcc)

# Function that retrieves the graphing arrays for either one of the three datasets that need graphs.
def getResultArr(res, test, ds):
    if ds in [1, 5, 6]:
        if ds == 1:
            out = ageDataset.getResultArr(test, res)
        if ds == 5:
            out = checkDatasets.getResultArr(test, res)
        if ds == 6:
            out = distanceDataset.getResultArr(test, res)
        return out

# Function that divides all values in the list of lists by ITERATIONS.
def divAll(total):
    for i in range(len(total)):
        for j in range(len(total[i])):
            for p in range(len(total[i][j])):
                for q in range(len(total[i][j][p])):
                    for r in range(len(total[i][j][p][q])):
                        total[i][j][p][q][r] /= ITERATIONS
    return total

# Function that adds up different ITERATIONS values one by one. (So not all at the same time).
def addToSum(total, add):
    for i in range(len(total)):
        for j in range(len(total[i])):
            for p in range(len(total[i][j])):
                for q in range(len(total[i][j][p])):
                    for r in range(len(total[i][j][p][q])):
                        total[i][j][p][q][r] += add[i][j][p][q][r]
    return total

# Function that prints all the graphs needed for the project.
# This could be made more general by perhaps making a single dataset file, but I did not do that.
def printGraphs(summedResultArrs):
    for i in range(len(summedResultArrs)):
        ageDataset.printGraph(summedResultArrs[i][0][0], "Age" + LEARNING_SYSTEMS[i] + "SFtrained")
        ageDataset.printGraph(summedResultArrs[i][0][1], "Age" + LEARNING_SYSTEMS[i] + "MFtrained")
        checkDatasets.printGraph(summedResultArrs[i][1][0], "Resource" + LEARNING_SYSTEMS[i] + "SFtrained")
        checkDatasets.printGraph(summedResultArrs[i][1][1], "Resource" + LEARNING_SYSTEMS[i] + "MFtrained")
        distanceDataset.printGraph(summedResultArrs[i][2][0], "Distance" + LEARNING_SYSTEMS[i] + "SFtrained")
        distanceDataset.printGraph(summedResultArrs[i][2][1], "Distance" + LEARNING_SYSTEMS[i] + "MFtrained")

if __name__ == "__main__":
    main()