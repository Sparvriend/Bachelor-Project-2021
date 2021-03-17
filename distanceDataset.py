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

def main():
    print("Generating training/testing data for the neural network")
    # Generating testing data, testing data points *2 because it is altered in its entirity later
    distanceDataset = dataGen.modifyData(addFail(dataGen.generatePerfectData(TESTING_DATA_POINTS*2)), TESTING_DATA_POINTS, 'TEST', 'Data/distance/distanceDatasetPreprocessed')

    # Generating training data
    singleFailTrain = dataGen.initData(1, TRAINING_DATA_POINTS, 'TRAIN', 'Data/distance/distanceDatasetPreprocessed1')

    # Training/testing with the neural net
    print("Training on single fail, testing on distance dataset")
    sfPredictions = setNN.testDataset(singleFailTrain, distanceDataset, MAX_ITERATIONS)

    # Graphing results
    print("Making graph for the distance dataset")
    makeDistanceGraph(distanceDataset, sfPredictions, "singleFailDistance - Figure 4")

# Resetting the age to a value which can fail
def addFail(df):
    for i in range(len(df.Age)):
        df.loc[i, "Distance"] = random.choice(list(range(0, 101)))

    return df

def makeDistanceGraph(test, predictions, name):
    legend = ["out-patients", "in-patients"]

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
        
        # Averaging the prediction results over the 3 nets
        resultArr = [i/3 for i in resultArr]
        plt.grid()
        if p == 0:
            plt.plot(list(range(0, 101)), resultArr, color = 'red', linewidth = 1.0)
        else:
            plt.plot(list(range(0, 101)), resultArr, '--', color = 'blue', linewidth = 1.0)

    plt.legend(legend)
    plt.ylabel('Output') 
    plt.xlabel('Distance')
    plt.grid()
    plt.title(name)
    plt.savefig('Data/distance/' + name + '.png')
    plt.clf()

if __name__ == "__main__":
    main()