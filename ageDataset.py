import pandas as pd
import dataGen
import setNN
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TESTING_DATA_POINTS = 1000
TRAINING_DATA_POINTS = 1200
MAX_ITERATIONS = 5000

# Age data set generation Python file made for the Bachelor Project by Timo Wahl (s3812030)

# TODO:
# - Should this be 4400 datapoints that all fail on the age condition
# Or 2200 datapoints that fail on the age condition and 2200 datapoints that are perfect?
# - Average men/women data over the three nets for figure 3? Or only over the first two nets? Or one of the three? 
# - Are the graphs correct? (Correct enough?)

def main():
    print("Generating training/testing data for the neural network")
    dfPerfect = dataGen.generatePerfectData(TESTING_DATA_POINTS)
    ageDataset = failAgeCondition(dfPerfect)
    dfPerfect = dataGen.generatePerfectData(TESTING_DATA_POINTS)

    singleFailTrain = dataGen.generateFailData(1, TRAINING_DATA_POINTS, 'TRAIN', 'Data/age/ageDatasetPreprocessed')
    multipleFailTrain = dataGen.generateFailData(6, TRAINING_DATA_POINTS, 'TRAIN', 'Data/age/ageDatasetPreprocessed')
    
    ageDataset = pd.concat([dfPerfect, ageDataset], axis = 0, ignore_index=True)
    ageDataset = editDataset(ageDataset)

    print("Training on multiple fail, testing on age dataset")
    mfPredictions = setNN.testDataset(multipleFailTrain, ageDataset, MAX_ITERATIONS)
    print("Training on single fail, testing on age dataset")
    sfPredictions = setNN.testDataset(singleFailTrain, ageDataset, MAX_ITERATIONS)

    print("Making graphs for the age dataset")
    makeAgeGraphSplit(ageDataset, mfPredictions, "MultipleFailTrainSplit - Figure 2")
    makeAgeGraphFullSplit(ageDataset, sfPredictions, "SingleFailTrainFullSplit - Figure 3")

def editDataset(df):
    df = dataGen.checkEligibility(df)
    tf = df[0]

    # dataGen.calcEligibilityPerc(tf)
    # dataGen.printFailOn(df[1], 1, TESTING_DATA_POINTS)

    # Shuffeling the dataframe
    tf = tf.sample(frac = 1).reset_index(drop=True)

    # Adding noise variables
    dfNoise = dataGen.generateNoiseData(TESTING_DATA_POINTS)
    tf = pd.concat([tf, dfNoise], axis = 1)

    # Printing to excel, preprocessing, then again printing to excel
    #tf.to_excel('Data/age/ageDataset.xlsx')
    tf = dataGen.preprocessData(tf)
    tf.to_excel('Data/age/ageDatasetPreProcessed.xlsx')

    return tf

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

def makeAgeGraphFullSplit(test, predictions, name):
    legend = ["Women", "Men"]

    for p in [0,1]:
        countArr = [0 for i in range(101)]; countArr = countArr[::5]
        eligArr = [0 for i in range(101)]; eligArr = eligArr[::5]
        resultArr = [0 for i in range(101)]; resultArr = resultArr[::5]
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
        
        resultArr = [i/3 for i in resultArr]
        plt.grid()
        if p == 0:
            plt.plot(list(range(0, 105, 5)), resultArr, '--', color = 'red', linewidth = 1.0)
        else:
            plt.plot(list(range(0, 105, 5)), resultArr, color = 'blue', linewidth = 1.0)

    plt.legend(legend)
    plt.ylabel('Output') 
    plt.xlabel('Age')
    plt.grid()
    plt.title(name)
    plt.savefig('Data/age/' + name + '.png')
    plt.clf()

def makeAgeGraphSplit(test, predictions, name):
    legend = ["1 Hidden Layer - men", "1 Hidden Layer - women", "2 Hidden Layers - men", "2 Hidden Layers - women", "3 Hidden Layers - men", "3 Hidden Layers - women"]
    colours = ['green', 'red', 'blue']

    for p in [0,1]:
        for j, predict in enumerate(predictions):
            countArr = [0 for i in range(101)]; countArr = countArr[::5]
            eligArr = [0 for i in range(101)]; eligArr = eligArr[::5]
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

    plt.legend(legend)
    plt.ylabel('Output') 
    plt.xlabel('Age')
    plt.grid()
    plt.title(name)
    plt.savefig('Data/age/' + name + '.png')
    plt.clf()

# Obsolete function
def makeAgeGraph(test, predictions, name):
    legend = ["1 Hidden Layer", "2 Hidden layers", "3 Hidden Layers"]
    colours = ['green', 'red', 'blue']

    for j, predict in enumerate(predictions):
        # Creating a tuple list, which keeps track of what eligibility score it was given and the amount of times an age has occured
        countArr = [0 for i in range(101)]; countArr = countArr[::5]
        eligArr = [0 for i in range(101)]; eligArr = eligArr[::5]
        resultArr = []

        for i in range(len(test.Age)):
            age = test.loc[i, "Age"]
            eligArr[int(age/5)] += predict[i]
            countArr[int(age/5)] += 1
        
        for i in range(len(eligArr)):
            if countArr[i] == 0:
                resultArr.append(0)
            else:
                resultArr.append(eligArr[i]/countArr[i])

        plt.grid()
        plt.plot(list(range(0, 105, 5)), resultArr, color = colours[j], linewidth = 1.0)

    plt.legend(legend)
    plt.ylabel('Output') 
    plt.xlabel('Age')
    plt.grid()
    plt.title(name)
    plt.savefig('Data/age/' + name + '.png')
    plt.clf()

if __name__ == "__main__":
    main()