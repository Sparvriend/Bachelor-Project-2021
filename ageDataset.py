import pandas as pd
import dataGen
import setNN
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TESTING_DATA_POINTS = 1000
TRAINING_DATA_POINTS = 1200
MAX_ITERATIONS = 3000

# Age data set generation Python file made for the Bachelor Project by Timo Wahl (s3812030)

# TODO:
# - Average men/women data over the three nets for figure 3? Or only over the first two nets? Or one of the three? 
# - Are the graphs correct? (Correct enough?)

def main():
    print("Generating training/testing data for the neural network")

    # Testing dataset with 50/50 split
    # ageDataset = failAgeCondition(dataGen.generatePerfectData(TESTING_DATA_POINTS))

    # Testing dataset with 37.5% split as is hinted at in the paper
    ageDataset = generateNewAge(dataGen.generatePerfectData(TESTING_DATA_POINTS))
    ageDataset = pd.concat([dataGen.generatePerfectData(TESTING_DATA_POINTS), ageDataset], axis = 0, ignore_index=True)
    ageDataset = dataGen.modifyData(ageDataset, TESTING_DATA_POINTS, 'TEST', 'Data/age/ageDatasetPreProcessed')

    # Generating the two training datasets
    singleFailTrain = dataGen.initData(1, TRAINING_DATA_POINTS, 'TRAIN', 'Data/age/ageDatasetPreprocessed1')
    multipleFailTrain = dataGen.initData(6, TRAINING_DATA_POINTS, 'TRAIN', 'Data/age/ageDatasetPreprocessed6')

    # Testing
    print("Training on multiple fail, testing on age dataset")
    mfPredictions = setNN.testDataset(multipleFailTrain, ageDataset, MAX_ITERATIONS)
    print("Training on single fail, testing on age dataset")
    sfPredictions = setNN.testDataset(singleFailTrain, ageDataset, MAX_ITERATIONS)

    # Making the graphs
    print("Making graphs for the age dataset")
    makeAgeGraphSplit(ageDataset, mfPredictions, "MultipleFailTrainSplit - Figure 2")
    makeAgeGraphFullSplit(ageDataset, sfPredictions, "SingleFailTrainFullSplit - Figure 3")

def generateNewAge(df):
    for i in range(len(df.Age)):
        df.loc[i, "Age"] = random.choice(list(range(0, 100, 5)))
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

# Making graphs for the age predictions, which are fully split in the paper between gender
def makeAgeGraphFullSplit(test, predictions, name):
    legend = ["Women", "Men"]

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
        else:
            plt.plot(list(range(0, 105, 5)), resultArr, color = 'blue', linewidth = 1.0)

    finalizeGraph(legend, name)

def makeAgeGraphSplit(test, predictions, name):
    legend = ["1 Hidden Layer - women", "2 Hidden Layers - women", "3 Hidden Layers - women", "1 Hidden Layer - men", "2 Hidden Layers - men", "3 Hidden Layers - men"]
    colours = ['green', 'red', 'blue']

    for p in [0,1]:
        for j, predict in enumerate(predictions):
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
                plt.plot(list(range(0, 105, 5)), resultArr, '--', color = colours[j], linewidth = 1.0)
            else:
                plt.plot(list(range(0, 105, 5)), resultArr, color = colours[j], linewidth = 1.0)

    finalizeGraph(legend, name)

def finalizeGraph(legend, name):
    plt.legend(legend)
    plt.ylabel('Output') 
    plt.xlabel('Age')
    plt.grid()
    plt.title(name)
    plt.savefig('Data/age/' + name + '.png')
    plt.clf()

if __name__ == "__main__":
    main()