import pandas as pd
import dataGen
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def getData(DATA_POINTS):
    print("Generating training/testing data for the neural network for the distance dataset")
    datasets = []
    datasets.append(dataGen.initData(1, DATA_POINTS, 'TRAIN', 'DataRes/distance/distanceDataset1'))
    datasets.append(dataGen.initData(6, DATA_POINTS, 'TRAIN', 'DataRes/distance/distanceDataset6'))
    datasets.append(dataGen.modifyData(failDistance(dataGen.generatePerfectData(DATA_POINTS*2)), DATA_POINTS, 'TEST', 'DataRes/distance/distanceDataset'))

    return datasets

def failDistance(df):
    for i in range(len(df.Age)):
        df.loc[i, "Distance"] = random.choice(list(range(0, 101)))
        if (df.loc[i, "InOut"] == 0 and df.loc[i, "Distance"] <= 50) or (df.loc[i, "InOut"] == 1 and df.loc[i, "Distance"] > 50):
            df.loc[i, "Eligible"] = 0
    return df

def printGraph(test, predictions, name):
    for p in [0,1]:
        countArr = [0 for i in range(101)]
        eligArr = countArr.copy()
        resultArr = countArr.copy()

        for j, predict in enumerate(predictions):
            for i in range(len(test.Age)):
                if test.loc[i, "InOut"] == p: 
                    distance = test.loc[i, "Distance"]
                    eligArr[distance] += predict[i]
                    countArr[distance] += 1
            
            for i in range(len(eligArr)):
                if countArr[i] == 0:
                    continue
                else:
                    resultArr[i] += eligArr[i]/countArr[i]
        
        plt.grid()
        if p == 0:
            plt.plot(list(range(0, 101)), resultArr, color = 'red', linewidth = 1.0)
        else:
            plt.plot(list(range(0, 101)), resultArr, '--', color = 'blue', linewidth = 1.0)

    plt.legend(["out-patients", "in-patients"])
    plt.ylabel('Output') 
    plt.xlabel('Distance')
    plt.grid()
    plt.savefig('DataRes/distance/' + name + '.png')
    plt.clf()