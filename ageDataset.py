import pandas as pd
import dataGen
import setNN
import random

DATA_POINTS = 2200

# Should this be 4400 datapoints that all fail on the age condition
# Or 2200 datapoints that fail on the age condition and 2200 datapoints that are perfect?

def main():
    dfPerfect = dataGen.generatePerfectData(DATA_POINTS)
    ageDataset = failAgeCondition(dfPerfect)
    dfPerfect = dataGen.generatePerfectData(DATA_POINTS)
    
    ageDataset = pd.concat([dfPerfect, ageDataset], axis = 0, ignore_index=True)
    ageDataset = editDataset(ageDataset)

    print("Testing with age dataset")
    setNN.testDataset(ageDataset)

def editDataset(df):
    df = dataGen.checkEligibility(df)
    tf = df[0]

    dataGen.calcEligibilityPerc(tf)
    dataGen.printFailOn(df[1], 1, DATA_POINTS)

    # Shuffeling the dataframe
    tf = tf.sample(frac = 1).reset_index(drop=True)

    # Adding noise variables
    dfNoise = dataGen.generateNoiseData(DATA_POINTS)
    tf = pd.concat([tf, dfNoise], axis = 1)

    # Printing to excel, preprocessing, then again printing to excel
    tf.to_excel('Data/age/ageDataset.xlsx')
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

if __name__ == "__main__":
    main()