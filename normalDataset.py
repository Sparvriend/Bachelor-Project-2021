import pandas as pd
import dataGen
import mlp

DATA_POINTS = 1000

# Normal data set generation Python file made for the Bachelor Project by Timo Wahl (s3812030)

def main():
    dat = getData()
    runMlp(dat, 1000)
    # runXGBoost()
    # runRandomForest()

def getData():
    print("Generating training/testing data for the neural network for the normal dataset")
    singleFailTrain = dataGen.initData(1, DATA_POINTS, 'TRAIN', 'DataRes/normal/normalDatasetPreProcessed1')
    singleFailTest = dataGen.initData(1, DATA_POINTS, 'TEST', 'DataRes/normal/normalDatasetPreProcessed1')
    multipleFailTrain = dataGen.initData(8, DATA_POINTS, 'TRAIN', 'DataRes/normal/normalDatasetPreProcessed8')
    multipleFailTest = dataGen.initData(8, DATA_POINTS, 'TEST', 'DataRes/normal/normalDatasetPreProcessed8')

    return (singleFailTrain, singleFailTest, multipleFailTrain, multipleFailTest)

def runMlp(datasets, maxIterations):
    print("Running the MLP learning system on the normal dataset")
    # Getting the results from training/testing on the data with the mlp
    print("Training on multiple fail, testing on multiple fail"); acc1 = mlp.testDataset(datasets[2], datasets[3], maxIterations)[1]
    print("Training on multiple fail, testing on single fail"); acc2 = mlp.testDataset(datasets[2], datasets[1], maxIterations)[1]
    print("Training on single fail, testing on multiple fail"); acc3 = mlp.testDataset(datasets[0], datasets[3], maxIterations)[1]
    print("Training on single fail, testing on single fail"); acc4 = mlp.testDataset(datasets[0], datasets[1], maxIterations)[1]

if __name__ == "__main__":
    main()