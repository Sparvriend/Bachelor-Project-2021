import pandas as pd
import dataGen
import setNN

def main():
    print("Generating training/testing data for the neural network")
    singleFailTrain = generateFailData(1, 1200, 'TRAIN')
    singleFailTest = generateFailData(1, 1000, 'TEST')
    multipleFailTrain = generateFailData(6, 1200, 'TRAIN')
    multipleFailTest = generateFailData(6, 1000, 'TEST')
    
    print("Training on multiple fail, testing on multiple fail")
    setNN.testDataset(multipleFailTrain, multipleFailTest)
    print("Training on multiple fail, testing on single fail")
    setNN.testDataset(multipleFailTrain, singleFailTest)
    print("Training on single fail, testing on multiple fail")
    setNN.testDataset(singleFailTrain, multipleFailTest)
    print("Training on single fail, testing on single fail")
    setNN.testDataset(singleFailTrain, singleFailTest)

# This function is used to generate the two datasets; one consisting of perfect datapoints
# and the other consisting of imperfect datapoints (for eligbility)
# As a parameter, the maximum amount of conditions that can be failed on are given to the function
# For the singular fail condition this is 1, while it is 6 for the multiple fail condition
# The function also calculates the eligibility percentage (should be 50%)
def generateFailData(MAX_FAIL_CONDITIONS, DATA_POINTS, TYPE):
    dfPerfect = dataGen.generatePerfectData(DATA_POINTS)
    dfSingleFail = dataGen.generatePerfectData(DATA_POINTS)
    dfSingleFail = dataGen.failConditions(dfSingleFail, MAX_FAIL_CONDITIONS)

    tf = pd.concat([dfPerfect, dfSingleFail], axis = 0, ignore_index=True)
    eligibilityResult = dataGen.checkEligibility(tf)
    tf = eligibilityResult[0]

    #dataGen.calcEligibilityPerc(tf)
    #dataGen.printFailOn(eligibilityResult[1], MAX_FAIL_CONDITIONS, DATA_POINTS)

    # Shuffeling the dataframe
    tf = tf.sample(frac = 1).reset_index(drop=True)

    # Adding noise variables
    dfNoise = dataGen.generateNoiseData(DATA_POINTS)
    tf = pd.concat([tf, dfNoise], axis = 1)

    tf = dataGen.preprocessData(tf)
    tf.to_excel('Data/normal/normalDatasetPreProcessed' + str(MAX_FAIL_CONDITIONS) + str(TYPE) + '.xlsx')

    return tf

if __name__ == "__main__":
    main()