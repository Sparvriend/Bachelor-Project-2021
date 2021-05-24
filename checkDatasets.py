import dataGen
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# This file is used to generate data for either one of the 4 conditions that were not included in the original Bench-Capon paper.
# It can also be used to retrieve the result array for the resource graph. Which can then be graphed with the printGraph function.

def getOnlyTest(DATA_POINTS, rule):
    if rule == 2:
        return dataGen.modifyData(failContribution(dataGen.generatePerfectData(DATA_POINTS*2)), DATA_POINTS, 'TEST', 'DataRes/contribution/contributionDataset')
    if rule == 3:
        return dataGen.modifyData(failCondition(dataGen.generatePerfectData(DATA_POINTS*2), "Spouse"), DATA_POINTS, 'TEST', 'DataRes/spouse/spouseDataset')
    if rule == 4:
        return dataGen.modifyData(failCondition(dataGen.generatePerfectData(DATA_POINTS*2), "Residency"), DATA_POINTS, 'TEST', 'DataRes/residency/residencyDataset')
    if rule == 5:
        return dataGen.modifyData(failResource(dataGen.generatePerfectData(DATA_POINTS*2)), DATA_POINTS, 'TEST', 'DataRes/resource/resourceDataset')

def failContribution(df):
    for i in range(int(len(df.Age)/2)):
        df.loc[i, "Eligible"] = 0; contrYears = [0,0,0,0,0]
        yearsPaid = random.choice(list(range(0, 4)))

        for q in range(yearsPaid):
            contrYears[random.choice(list(range(0, 5)))] = 1
        for q in range(len(contrYears)):
            df.loc[i, "Contribution" + str(q+1)] = contrYears[q]
    return df

def failCondition(df, condition):
    for i in range(len(df.Age)):
        df.loc[i, condition] = random.choice(list(range(0, 2)))
        if (df.loc[i, condition] == 0):
            df.loc[i, "Eligible"] = 0
    return df

def failResource(df):
    for i in range(int(len(df.Age)/2)):
        df.loc[i, "Resource"] = random.choice(list(range(3001, 10001, 10)))
        df.loc[i, "Eligible"] = 0
    return df

# This works, but only when the amount of datapoints is sufficiently high.
def getResultArr(test, prediction):
    countArr = np.zeros(10000); valueArr = np.zeros(10000)

    for i in range(len(test.Age)):
        valueArr[test.loc[i, "Resource"]] += prediction[i]
        countArr[test.loc[i, "Resource"]] += 1

    for i in range(len(valueArr)):
        if countArr[i] == 0:
                valueArr[i] = None
        else:
            valueArr[i] = valueArr[i]/countArr[i]

    return [valueArr]

def printGraph(valueArr, name):
    # Masking to cover missing values
    mask = np.isfinite(valueArr[0]); xs = np.arange(10000)
    plt.plot(xs[mask], valueArr[0][mask], color = 'red', linewidth = 1.0)
    plt.legend(["Resource"])
    plt.ylabel('Output') 
    plt.xlabel('Resource')
    plt.grid()
    plt.savefig('DataRes/resource/' + name + '.png')
    plt.clf()