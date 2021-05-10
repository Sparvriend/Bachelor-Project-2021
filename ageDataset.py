import pandas as pd
import dataGen
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def getData(DATA_POINTS):
    print("Generating training/testing data for the neural network for the age dataset")
    datasets = []
    datasets.append(dataGen.initData(1, DATA_POINTS, 'TRAIN', 'DataRes/age/ageDataset1'))
    datasets.append(dataGen.initData(6, DATA_POINTS, 'TRAIN', 'DataRes/age/ageDataset6'))
    datasets.append(dataGen.modifyData(failAge(dataGen.generatePerfectData(DATA_POINTS*2)), DATA_POINTS, 'TEST', 'DataRes/age/ageDataset'))

    return datasets

def failAge(df):
    for i in range(len(df.Age)):
        df.loc[i, "Age"] = random.choice(list(range(0, 105, 5)))
        if (df.loc[i, "Gender"] == 0 and df.loc[i, "Age"] < 60) or (df.loc[i, "Gender"] == 1 and df.loc[i, "Age"] < 65):
            df.loc[i, "Eligible"] = 0
    return df

# Making a graph for the age predictions, which are fully split in the paper between gender
def printGraph(test, predictions, name):
    for p in [0,1]:
        countArr = [0 for i in range(101)]; countArr = countArr[::5]
        eligArr = countArr.copy()
        resultArr = countArr.copy()

        for j, predict in enumerate(predictions):
            for i in range(len(test.Age)):
                if test.loc[i, "Gender"] == p: 
                    age = test.loc[i, "Age"]
                    eligArr[int(age/5)] += predict[i]
                    countArr[int(age/5)] += 1
            
            for i in range(len(eligArr)):
                if countArr[i] == 0:
                    continue
                else:
                    resultArr[i] += eligArr[i]/countArr[i]
        
        plt.grid()
        if p == 0:
            plt.plot(list(range(0, 105, 5)), resultArr, '--', color = 'red', linewidth = 1.0)
        else:
            plt.plot(list(range(0, 105, 5)), resultArr, color = 'blue', linewidth = 1.0)

    plt.legend(["Women", "Men"])
    plt.ylabel('Output') 
    plt.xlabel('Age')
    plt.grid()
    plt.savefig('DataRes/age/' + name + '.png')
    plt.clf()