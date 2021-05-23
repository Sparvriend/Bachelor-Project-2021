import dataGen
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def getOnlyTest(DATA_POINTS):
    return dataGen.modifyData(failDistance(dataGen.generatePerfectData(DATA_POINTS*2)), DATA_POINTS, 'TEST', 'DataRes/distance/distanceDataset')

def failDistance(df):
    for i in range(len(df.Age)):
        df.loc[i, "Distance"] = random.choice(list(range(0, 101)))
        if (df.loc[i, "InOut"] == 0 and df.loc[i, "Distance"] <= 50) or (df.loc[i, "InOut"] == 1 and df.loc[i, "Distance"] > 50):
            df.loc[i, "Eligible"] = 0
    return df

def getResultArr(test, prediction):
    out = []
    for p in [0,1]:
        countArr = [0 for i in range(101)]
        eligArr = countArr.copy()
        resultArr = countArr.copy()

        for i in range(len(test.Age)):
            if test.loc[i, "InOut"] == p: 
                distance = test.loc[i, "Distance"]
                eligArr[distance] += prediction[i]
                countArr[distance] += 1
            
        for i in range(len(eligArr)):
            if countArr[i] == 0:
                continue
            else:
                resultArr[i] += eligArr[i]/countArr[i]
        out.append(resultArr)
    return out
        
def printGraph(resultArrs, name):
    plt.grid()
    plt.plot(list(range(0, 101)), resultArrs[0], color = 'red', linewidth = 1.0)
    plt.plot(list(range(0, 101)), resultArrs[1], '--', color = 'blue', linewidth = 1.0)
    plt.legend(["out-patients", "in-patients"])
    plt.ylabel('Output') 
    plt.xlabel('Distance')
    plt.grid()
    plt.savefig('DataRes/distance/' + name + '.png')
    plt.clf()