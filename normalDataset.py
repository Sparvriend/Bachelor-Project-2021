import dataGen

# This file can be used to generate either training or testing data for the models to be trained/tested on.
# The datasets are either single fail or multiple fail.

def getOnlyTrain(DATA_POINTS):
    print("Generating training data for the learning systems for the normal dataset")
    datasets = []
    datasets.append(dataGen.initData(1, DATA_POINTS, 'TRAIN', 'DataRes/normal/normalDatasetPreProcessed1'))
    datasets.append(dataGen.initData(6, DATA_POINTS, 'TRAIN', 'DataRes/normal/normalDatasetPreProcessed6'))
    return datasets

def getOnlyTest(DATA_POINTS):
    print("Generating testing data for the learning systems for the normal dataset")
    datasets = []
    datasets.append(dataGen.initData(1, DATA_POINTS, 'TEST', 'DataRes/normal/normalDatasetPreProcessed1'))
    datasets.append(dataGen.initData(6, DATA_POINTS, 'TEST', 'DataRes/normal/normalDatasetPreProcessed6'))
    return datasets