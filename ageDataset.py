import dataGen
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The eligibility for the age test dataset is not 37.5% currently
def getOnlyTest(DATA_POINTS):
    return dataGen.modifyData(failAge(dataGen.generatePerfectData(DATA_POINTS*2)), DATA_POINTS, 'TEST', 'DataRes/age/ageDataset')

def failAge(df):
    for i in range(len(df.Age)):
        df.loc[i, "Age"] = random.choice(list(range(0, 105, 5)))
        if (df.loc[i, "Gender"] == 0 and df.loc[i, "Age"] < 60) or (df.loc[i, "Gender"] == 1 and df.loc[i, "Age"] < 65):
            df.loc[i, "Eligible"] = 0
    return df

def getResultArr(test, prediction):
    out = []
    for p in [0,1]:
        countArr = [0 for i in range(101)]; countArr = countArr[::5]
        eligArr = countArr.copy()
        resultArr = countArr.copy()

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
    plt.plot(list(range(0, 105, 5)), resultArrs[0], '--', color = 'red', linewidth = 1.0)
    plt.plot(list(range(0, 105, 5)), resultArrs[1], color = 'blue', linewidth = 1.0)
    plt.legend(["Women", "Men"])
    plt.ylabel('Output') 
    plt.xlabel('Age')
    plt.grid()
    plt.savefig('DataRes/age/' + name + '.png')
    plt.clf()