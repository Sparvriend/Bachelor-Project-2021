import dataGen

# Normal data set generation Python file made for the Bachelor Project by Timo Wahl (s3812030)
def getData(DATA_POINTS):
    print("Generating training/testing data for the neural network for the normal dataset")
    singleFailTrain = dataGen.initData(1, DATA_POINTS, 'TRAIN', 'DataRes/normal/normalDatasetPreProcessed1')
    singleFailTest = dataGen.initData(1, DATA_POINTS, 'TEST', 'DataRes/normal/normalDatasetPreProcessed1')
    multipleFailTrain = dataGen.initData(8, DATA_POINTS, 'TRAIN', 'DataRes/normal/normalDatasetPreProcessed8')
    multipleFailTest = dataGen.initData(8, DATA_POINTS, 'TEST', 'DataRes/normal/normalDatasetPreProcessed8')

    return (singleFailTrain, singleFailTest, multipleFailTrain, multipleFailTest)