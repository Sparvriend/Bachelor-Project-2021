import dataGen
import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def getData(DATA_POINTS, rule):
    datasets = []
    
    if rule == 2:
        print("Generating training/testing data for the neural network for the contribution dataset")
        datasets.append(dataGen.initData(1, DATA_POINTS, 'TRAIN', 'DataRes/contribution/contributionDataset1'))
        datasets.append(dataGen.initData(6, DATA_POINTS, 'TRAIN', 'DataRes/contribution/contributionDataset6'))
        datasets.append(dataGen.modifyData(failContribution(dataGen.generatePerfectData(DATA_POINTS*2)), DATA_POINTS, 'TEST', 'DataRes/contribution/contributionDataset'))
    if rule == 3:
        print("Generating training/testing data for the neural network for the spouse dataset")
        datasets.append(dataGen.initData(1, DATA_POINTS, 'TRAIN', 'DataRes/spouse/spouseDataset1'))
        datasets.append(dataGen.initData(6, DATA_POINTS, 'TRAIN', 'DataRes/spouse/spouseDataset6'))
        datasets.append(dataGen.modifyData(failCondition(dataGen.generatePerfectData(DATA_POINTS*2), 'Spouse'), DATA_POINTS, 'TEST', 'DataRes/spouse/spouseDataset'))
    if rule == 4:
        print("Generating training/testing data for the neural network for the residency dataset")
        datasets.append(dataGen.initData(1, DATA_POINTS, 'TRAIN', 'DataRes/residency/residencyDataset1'))
        datasets.append(dataGen.initData(6, DATA_POINTS, 'TRAIN', 'DataRes/residency/residencyDataset6'))
        datasets.append(dataGen.modifyData(failCondition(dataGen.generatePerfectData(DATA_POINTS*2), 'Residency'), DATA_POINTS, 'TEST', 'DataRes/residency/residencyDataset'))
    if rule == 5:
        print("Generating training/testing data for the neural network for the resource dataset")
        datasets.append(dataGen.initData(1, DATA_POINTS, 'TRAIN', 'DataRes/resource/resourceDataset1'))
        datasets.append(dataGen.initData(6, DATA_POINTS, 'TRAIN', 'DataRes/resource/resourceDataset6'))
        datasets.append(dataGen.modifyData(failResource(dataGen.generatePerfectData(DATA_POINTS*2)), DATA_POINTS, 'TEST', 'DataRes/resource/resourceDataset'))

    return datasets

def failContribution(df):
    for i in range(int(len(df.Age)/2)):
        df.loc[i, "Eligible"] = 0
        contrYears = [0,0,0,0,0]
        yearsPaid = random.choice(list(range(0, 3)))

        for q in range(yearsPaid):
            contrYears[random.choice(list(range(0, 5)))] = 1
        for q in range(len(contrYears)):
            df.loc[i, "Contribution" + str(q+1)] = contrYears[q]
    return df

def failCondition(df, condition):
    for i in range(len(df.Age)):
        df.loc[i, condition] = random.choice(list(range(0, 1)))
        if (df.loc[i, condition] == 0):
            df.loc[i, "Eligible"] = 0
    return df

def failResource(df):
    for i in range(len(df.Age)):
        df.loc[i, "Resource"] = random.choice(list(range(0, 6000)))
        if (df.loc[i, "Resource"] > 3000):
            df.loc[i, "Eligible"] = 0
    return df

# Function that sums the individual accuracies of the models
def printBoolean(test, predictions, name):
    value = 0; counter = 0

    for j, predict in enumerate(predictions):
        for i in range(len(test.Age)):
            counter += 1
            if predict[i] == test.loc[i, "Eligible"]:
                value += 1
    print("Accuracy value for the boolean rule " + name + " = " + str(value/counter*100) + " %")

# This needs work, because its not working correctly
def printNumericalGraph(test, predictions, name):
    countArr = np.zeros(6000); valueArr = np.zeros(6000)

    for j, predict in enumerate(predictions):
        for i in range((len(test.Age))):
            valueArr[test.loc[i, "Resource"]] += predict[i]
            countArr[test.loc[i, "Resource"]] += 1

    for i in range(len(valueArr)):
        if countArr[i] == 0:
                valueArr[i] = None
        else:
            valueArr[i] = valueArr[i]/countArr[i]

    # Masking to cover missing values
    mask = np.isfinite(valueArr); xs = np.arange(6000)
    plt.plot(xs[mask], valueArr[mask], color = 'red', linewidth = 1.0)
    
    plt.legend(["Resource"])
    plt.ylabel('Output') 
    plt.xlabel('Resource')
    plt.grid()
    plt.savefig('DataRes/resource/' + name + '.png')
    plt.clf()