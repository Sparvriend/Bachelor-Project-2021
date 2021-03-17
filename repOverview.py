import pandas as pd
import distanceDataset
import normalDataset
import ageDataset

iterations = 20

# TODO:
# - implement age and distance dataset accuracy/graph averaging

def main():
    runNormalDataset()
    #ageDataset.main()
    #distanceDataset.main()

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