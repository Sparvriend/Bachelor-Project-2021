import pandas as pd
import dataGen
import setNN

TESTING_DATA_POINTS = 1000
TRAINING_DATA_POINTS = 1200
MAX_ITERATIONS = 3000

# Normal data set generation Python file made for the Bachelor Project by Timo Wahl (s3812030)

def main():
    print("Generating training/testing data for the neural network")
    singleFailTrain = dataGen.initData(1, TRAINING_DATA_POINTS, 'TRAIN', 'Data/normal/normalDatasetPreProcessed1')
    singleFailTest = dataGen.initData(1, TESTING_DATA_POINTS, 'TEST', 'Data/normal/normalDatasetPreProcessed1')
    multipleFailTrain = dataGen.initData(6, TRAINING_DATA_POINTS, 'TRAIN', 'Data/normal/normalDatasetPreProcessed6')
    multipleFailTest = dataGen.initData(6, TESTING_DATA_POINTS, 'TEST', 'Data/normal/normalDatasetPreProcessed6')
    
    print("Training on multiple fail, testing on multiple fail")
    setNN.testDataset(multipleFailTrain, multipleFailTest, MAX_ITERATIONS)
    print("Training on multiple fail, testing on single fail")
    setNN.testDataset(multipleFailTrain, singleFailTest, MAX_ITERATIONS)
    print("Training on single fail, testing on multiple fail")
    setNN.testDataset(singleFailTrain, multipleFailTest, MAX_ITERATIONS)
    print("Training on single fail, testing on single fail")
    setNN.testDataset(singleFailTrain, singleFailTest, MAX_ITERATIONS)

if __name__ == "__main__":
    main()