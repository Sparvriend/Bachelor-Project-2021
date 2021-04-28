import dataGen

# Normal data set generation Python file made for the Bachelor Project by Timo Wahl (s3812030)
def getData(DATA_POINTS):
    print("Generating training/testing data for the neural network for the normal dataset")
    datasets = []
    datasets.append(dataGen.initData(1, DATA_POINTS, 'TRAIN', 'DataRes/normal/normalDatasetPreProcessed1'))
    datasets.append(dataGen.initData(1, DATA_POINTS, 'TEST', 'DataRes/normal/normalDatasetPreProcessed1'))
    datasets.append(dataGen.initData(6, DATA_POINTS, 'TRAIN', 'DataRes/normal/normalDatasetPreProcessed6'))
    datasets.append(dataGen.initData(6, DATA_POINTS, 'TEST', 'DataRes/normal/normalDatasetPreProcessed6'))

    return datasets