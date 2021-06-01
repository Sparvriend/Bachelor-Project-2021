import pandas as pd
import distanceDataset
import normalDataset
import ageDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from datetime import timedelta

iterations = 2
TESTING_DATA_POINTS = 2000
TRAINING_DATA_POINTS = 2400

# Replication overview Python file made for the Bachelor Project by Timo Wahl (s3812030)
# This file is used to average over multiple data set runs and print the accuracies/graphs over those runs

# TODO:
# - implement age graph averaging (runAgeDataset())

def main():
    startTime = time.time()
    #runNormalDataset()
    #runDistanceDataset()
    #runAgeDataset()
    print("Time elapsed: " + str(timedelta(seconds = time.time() - startTime)))

def getAvgAccuracy(sf, mf, type):
    specPrints = ["Single layer neural network result: ", "Double layer neural network result: ", "Triple layer neural network result: "]
    typeOfNN = ["Trained on single fail, tested on " + str(type), "Trained on multiple fail, tested on " + str(type)]

    final = [[[],[],[]],[[],[],[]]]
    for i in range(iterations):
        for j in range(len(sf[i])):
            final[0][j].append(sf[i][j])
            final[1][j].append(mf[i][j])

    print("ITERATIONS: " + str(iterations))
    for i, NNtype in enumerate(final):
        print("==================================================================================")
        print(typeOfNN[i])
        for j, net in enumerate(NNtype):
            print(specPrints[j] + str(round(float(np.mean(net))*100, 4)) + " " + str(round(float(np.std(net))*100, 4)))

# Function that runs the age dataset multiple times
# It averages the results over the amount of iterations and then prints the result
# This function is pure crap
def runAgeDataset():
    neuralNets = ["1 Hidden Layer", "2 Hidden Layers", "3 Hidden Layers"]
    colours = ['red', 'blue']
    legend = ['Women', 'Men']
    accTotalSF = []
    accTotalMF = []

    graphArr = np.zeros((2,3,21)).tolist(); graphArrFS = np.zeros((2,21)).tolist()

    for i in range(iterations):
        result = ageDataset.main()
        accTotalSF.append(result[4])
        accTotalMF.append(result[5])
        fullSplit = []; fullSplit.append(result[2]); fullSplit.append(result[3])
        for j in range(3):
            for p in [0,1]:
                out = result[p][j]
                for q in range(len(out[0])):
                    graphArr[p][j][q] += out[0][q]
        
        for q in range(len(fullSplit[0])):
            graphArrFS[0][q] += fullSplit[0][q]
            graphArrFS[1][q] += fullSplit[1][q]

    for j in range(3):
        for p in [0,1]:
            out = result[p][j]
            for q in range(len(graphArr[p][j])):
                graphArr[p][j][q] /= iterations
        
        plt.plot(list(range(0, 105, 5)), graphArr[0][j], '--', color = colours[0], linewidth = 1.0)
        plt.plot(list(range(0, 105, 5)), graphArr[1][j], color = colours[1], linewidth = 1.0)
        ageDataset.finalizeGraph(legend, "Multiple fail train, age test, " + str(j+1) + " hidden layers over " + str(iterations) + " iterations")

    for q in range(len(graphArrFS[0])):
        graphArrFS[0][q] /= iterations
        graphArrFS[1][q] /= iterations
    
    plt.plot(list(range(0, 105, 5)), graphArrFS[0], '--', color = colours[0], linewidth = 1.0)
    plt.plot(list(range(0, 105, 5)), graphArrFS[1], color = colours[1], linewidth = 1.0)
    ageDataset.finalizeGraph(legend, "Single fail train, age test, all nets over " + str(iterations) + " iterations")

    getAvgAccuracy(accTotalSF, accTotalMF, "Age")

# Function that runs the distance dataset multiple times
# It averages the results over the amount of iterations and then prints the result
def runDistanceDataset():
    legend = ["out-patients", "in-patients"]
    name = ["Single fail train, distance test, averaged over " + str(iterations) + " iterations", "Multiple fail train, distance test, averaged over " + str(iterations) + " iterations"]
    totalResult = []
    accTotalSF = []
    accTotalMF = []

    for i in range(iterations):
        totalResult.append(distanceDataset.main())
        accTotalSF.append(totalResult[i][3])
        accTotalMF.append(totalResult[i][4])
    getAvgAccuracy(accTotalSF, accTotalMF, "Distance")

    for q in range(2):
        countArrOut = [0 for i in range(101)]; eligArrOut = countArrOut.copy(); resultArrOut = countArrOut.copy()
        countArrIn = countArrOut.copy(); eligArrIn = countArrOut.copy(); resultArrIn = countArrOut.copy()

        for i in range(iterations):
            res = totalResult[i]; predictions = res[q]; test = res[2]
            for j, predict in enumerate(predictions):
                for p in range(len(test.Age)):
                    distance = test.loc[p, "Distance"]
                    if test.loc[p, "InOut"] == 0: 
                        eligArrOut[distance] += predict[p]
                        countArrOut[distance] += 1
                    if test.loc[p, "InOut"] == 1: 
                        eligArrIn[distance] += predict[p]
                        countArrIn[distance] += 1

        for i in range(len(eligArrIn)):
            resultArrIn[i] = eligArrIn[i]/countArrIn[i]
            resultArrOut[i] = eligArrOut[i]/countArrOut[i]
        
        plt.plot(list(range(0, 101)), resultArrOut, color = 'red', linewidth = 1.0)
        plt.plot(list(range(0, 101)), resultArrIn, '--', color = 'blue', linewidth = 1.0)
        distanceDataset.finalizeGraph(legend, name[q])

# Function that runs the normal dataset multiple times
def runNormalDataset():
    final = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
    genPrints = ["Training on multiple fail, testing on multiple fail final result", "Training on multiple fail, testing on single fail final result",
    "Training on single fail, testing on multiple fail final result", "Training on single fail, testing on single fail final result"]
    specPrints = ["Single layer neural network result: ", "Double layer neural network result: ", "Triple layer neural network result: "]

    # Adding up accuracies over iterations
    for i in range(iterations):
        print("iteration " + str(i))
        accuracies = normalDataset.main()
        for j, accuracy in enumerate(accuracies):
            for p, prediction in enumerate(accuracy):
                final[j][p].append(prediction)

    # Averaging over iterations
    print("ITERATIONS: " + str(iterations))
    for i in range(len(final)):
        print("==================================================================================")
        print(genPrints[i])
        for j in range(len(final[i])):
            print(specPrints[j] + str(round(float(np.mean(final[i][j]))*100, 4)) + " " + str(round(float(np.std(final[i][j]))*100, 4)))

if __name__ == "__main__":
    main()