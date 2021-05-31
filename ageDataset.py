import dataGen
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# This file is used to generate data for the age dataset. Besides doing that, it can also be used to retrieve,
# a result array, which contains the values to be graphed, which can then be graphed with the printGraph function.

def getOnlyTest(DATA_POINTS):
    return dataGen.modifyData(failAge(dataGen.generatePerfectData(DATA_POINTS*2)), DATA_POINTS, 'TEST', 'DataRes/age/ageDataset')

def failAge(df):
    for i in range(int(len(df.Age)*0.625)):
        if df.loc[i, "Gender"] == 1:
            df.loc[i, "Age"] = random.choice(list(range(0, 65, 5)))
        if df.loc[i, "Gender"] == 0:
            df.loc[i, "Age"] = random.choice(list(range(0, 60, 5)))
        df.loc[i, "Eligible"] = 0
    return df

def getResultArr(test, prediction):
    out = []
    for p in [0,1]:
        countArr = [0 for i in range(101)]; countArr = countArr[::5]
        eligArr = countArr.copy(); resultArr = countArr.copy()

        for i in range(len(test.Age)):
            if test.loc[i, "Gender"] == p: 
                age = test.loc[i, "Age"]
                eligArr[int(age/5)] += prediction[i]
                countArr[int(age/5)] += 1
            
        for i in range(len(eligArr)):
            if countArr[i] == 0:
                continue
            else:
                resultArr[i] += eligArr[i]/countArr[i]
        out.append(resultArr)
    return out

def printGraph(resultArrs, name):
    plt.grid()
    plt.ylim(0.0, 1.05)
    plt.plot(list(range(0, 105, 5)), resultArrs[0], '--', color = 'red', linewidth = 1.0)
    plt.plot(list(range(0, 105, 5)), resultArrs[1], color = 'blue', linewidth = 1.0)
    plt.legend(["Women", "Men"])
    plt.ylabel('Output') 
    plt.xlabel('Age')
    plt.grid()
    plt.savefig('DataRes/age/' + name + '.png')
    plt.clf()