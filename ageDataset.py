import pandas as pd
import dataGen
import setNN
import random
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TESTING_DATA_POINTS = 1000
TRAINING_DATA_POINTS = 1200
MAX_ITERATIONS = 3000

# Age data set generation Python file made for the Bachelor Project by Timo Wahl (s3812030)

def main():

    # Testing dataset with 37.5% eligibility, as is hinted at in the paper
    print("Generating training/testing data for the neural network")
    ageDataset = dataGen.modifyData(generateNewAge(dataGen.generatePerfectData(TESTING_DATA_POINTS*2)), TESTING_DATA_POINTS, 'TEST', 'Data/age/ageDatasetPreProcessed')

    # Generating the two training datasets
    singleFailTrain = dataGen.initData(1, TRAINING_DATA_POINTS, 'TRAIN', 'Data/age/ageDatasetPreprocessed1')
    multipleFailTrain = dataGen.initData(6, TRAINING_DATA_POINTS, 'TRAIN', 'Data/age/ageDatasetPreprocessed6')

    # Testing on the neural networks
    print("Training on multiple fail, testing on age dataset"); resSF = setNN.testDataset(singleFailTrain, ageDataset, MAX_ITERATIONS)
    print("Training on single fail, testing on age dataset"); resMF = setNN.testDataset(multipleFailTrain, ageDataset, MAX_ITERATIONS)
    sfPredictions = resSF[0]; acc1 = resSF[1]
    mfPredictions = resMF[0]; acc2 = resMF[1]

    # Making the graphs, as from the paper (figure 2 and 3)
    print("Making graphs for the age dataset")
    femOutMF, MalOutMF = makeAgeGraphNNSplit(ageDataset, mfPredictions, "MultipleFailTrainSplit - Figure 2")
    femOutSF, MalOutSF = makeAgeGraphFullSplit(ageDataset, sfPredictions, "SingleFailTrainFullSplit - Figure 3")

    return (femOutMF, MalOutMF, femOutSF, MalOutSF, acc1, acc2)

# Function that generates a new age value, from 0-105, with steps of 5
def generateNewAge(df):
    for i in range(len(df.Age)):
        df.loc[i, "Age"] = random.choice(list(range(0, 105, 5)))
    return df

# Failing on age conditions with steps of 5 for the age
def failAgeCondition(df):
    for i in range(len(df.Age)):
        if df.loc[i, "Gender"] == "Male":
            df.loc[i, "Age"] = random.choice(list(range(0, 65, 5)))
            continue

        if df.loc[i, "Gender"] == "Female":
            df.loc[i, "Age"] = random.choice(list(range(0, 60, 5)))
            continue

    return df

# Making a graph for the age predictions, which are fully split in the paper between gender (figure 3)
def makeAgeGraphFullSplit(test, predictions, name):
    legend = ["Women", "Men"]; outF = []; outM = []

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
        
        # Dividing by 3, as the results for the three neural nets need to be averaged for the full split (?)
        resultArr = [i/3 for i in resultArr]
        plt.grid()
        if p == 0:
            plt.plot(list(range(0, 105, 5)), resultArr, '--', color = 'red', linewidth = 1.0)
            outF = resultArr
        else:
            plt.plot(list(range(0, 105, 5)), resultArr, color = 'blue', linewidth = 1.0)
            outM = resultArr

    finalizeGraph(legend, name)

    return (outF, outM)

# Making a graph for the three neural network results on the age dataset (figure 2, but split in three figures in this case)
def makeAgeGraphNNSplit(test, predictions, name):
    neuralNets = ["1 Hidden Layer", "2 Hidden Layers", "3 Hidden Layers"]; outM = np.zeros((3, 0)).tolist(); outF = np.zeros((3, 0)).tolist()
    colours = ['red', 'blue']
    legend = ['Women', 'Men']

    for j, predict in enumerate(predictions):
        for p in [0,1]:
            countArr = [0 for i in range(101)]; countArr = countArr[::5]
            eligArr = countArr.copy()
            resultArr = []

            for i in range(len(test.Age)):
                if test.loc[i, "Gender"] == p: 
                    age = test.loc[i, "Age"]
                    eligArr[int(age/5)] += predict[i]
                    countArr[int(age/5)] += 1
            
            for i in range(len(eligArr)):
                if countArr[i] == 0:
                    resultArr.append(0)
                else:
                    resultArr.append(eligArr[i]/countArr[i])

            plt.grid()
            if p == 0:
                plt.plot(list(range(0, 105, 5)), resultArr, '--', color = colours[p], linewidth = 1.0)
                outF[j].append(resultArr)
            else:
                plt.plot(list(range(0, 105, 5)), resultArr, color = colours[p], linewidth = 1.0)
                outM[j].append(resultArr)
        finalizeGraph(legend, name + " - " + str(neuralNets[j]))

    return (outF, outM)    

# Function that finalizes a graph (adding legend/labels etc)
def finalizeGraph(legend, name):
    plt.legend(legend)
    plt.ylabel('Output') 
    plt.xlabel('Age')
    plt.grid()
    #plt.title(name)
    plt.savefig('Data/age/' + name + '.png')
    plt.clf()

if __name__ == "__main__":
    main()