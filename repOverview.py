import pandas as pd
import distanceDataset
import normalDataset
import ageDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

iterations = 20

# TODO:
# - implement age graph averaging
# - implement option to remove third NN from running the distance dataset, as it has very bad performance (<60), but only sometimes?
# - Remove graph making/accuracy measuring from the dataset files, such that it can be used here when the itertions is set to 1.

def main():
    #runNormalDataset()
    runDistanceDataset()

def runDistanceDataset():
    legend = ["out-patients", "in-patients"]
    name = "singleFailDistance averaged over " + str(iterations) +  " iterations - Figure 4"
    countArrOut = [0 for i in range(101)]; eligArrOut = countArrOut.copy(); resultArrOut = countArrOut.copy()
    countArrIn = countArrOut.copy(); eligArrIn = countArrOut.copy(); resultArrIn = countArrOut.copy()

    for i in range(iterations):
        res = distanceDataset.main(); predictions = res[0]; test = res[1]
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
    
    plt.grid()
    plt.plot(list(range(0, 101)), resultArrOut, color = 'red', linewidth = 1.0)
    plt.plot(list(range(0, 101)), resultArrIn, '--', color = 'blue', linewidth = 1.0)
    distanceDataset.finalizeGraph(legend, name)

def runAgeDataset():

def runNormalDataset():
    final = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    genPrints = ["Training on multiple fail, testing on multiple fail final result", "Training on multiple fail, testing on single fail final result",
    "Training on single fail, testing on multiple fail final result", "Training on single fail, testing on single fail final result"]
    specPrints = ["Single layer neural network accuracy: ", "Double layer neural network accuracy: ", "Triple layer neural network accuracy: "]

    # Adding up accuracies over iterations
    for i in range(iterations):
        accuracies = normalDataset.main()
        for j, accuracy in enumerate(accuracies):
            for p, prediction in enumerate(accuracy):
                final[j][p] += prediction

    # Averaging over iterations
    print("ITERATIONS: " + str(iterations))
    for i in range(len(final)):
        print("==================================================================================")
        print(genPrints[i])
        for j in range(len(final[i])):
            final[i][j] /= iterations
            print(specPrints[j] + str(final[i][j]))

if __name__ == "__main__":
    main()