import pandas as pd
import random

# Constant for the in out patient distance, measured in km
IN_OUT_PATIENT_DISTANCE = 20
NOISE_VARIABLES = 52
    
# Data generation Python file made for the Bachelor Project by Timo Wahl (s3812030)
# The paper that is constantly mentioned in this file:
# https://dl.acm.org/doi/abs/10.1145/158976.159012?casa_token=cTqiK-PMwnEAAAAA:KtSh_D8f5J3cV4sqSH3qyKG-XhHAb28hNt0au51BNDl4VdSQQ6aKp1W_baNu2aJ6O7LPL1YbOhhX

# TODO:
# - Add data generation for the two extra datasets for Age and Distance (mentioned in paper)
# - TypeA = multiple fail, typeB = single fail
# - Code in PEP8

def main():
    
    # Generating data for multiple and single fail conditions
    # The input to the neural network is a 64x2400 matrix (Adding eligiblity makes it a 65x2400 matrix)
    dfSF = generateFailData(1, 1200)
    dfMF = generateFailData(6, 1200)

# This function is used to generate the two datasets; one consisting of perfect datapoints
# and the other consisting of imperfect datapoints (for eligbility)
# As a parameter, the maximum amount of conditions that can be failed on are given to the function
# For the singular fail condition this is 1, while it is 6 for the multiple fail condition
# The function also calculates the eligibility percentage (should be 50%)
def generateFailData(MAX_FAIL_CONDITIONS, DATA_POINTS):
    dfPerfect = generatePerfectData(DATA_POINTS)
    dfSingleFail = generatePerfectData(DATA_POINTS)
    dfSingleFail = failConditions(dfSingleFail, MAX_FAIL_CONDITIONS)

    tf = pd.concat([dfPerfect, dfSingleFail], axis = 0, ignore_index=True)
    eligibilityResult = checkEligibility(tf)
    tf = eligibilityResult[0]

    #calcEligibilityPerc(tf)
    #printFailOn(eligibilityResult[1], MAX_FAIL_CONDITIONS, DATA_POINTS)

    # Shuffeling the dataframe
    tf = tf.sample(frac = 1).reset_index(drop=True)

    # Adding noise variables
    dfNoise = generateNoiseData(DATA_POINTS)
    tf = pd.concat([tf, dfNoise], axis = 1)

    # Printing to excel, preprocessing, then again printing to excel
    tf.to_excel('tf' + str(MAX_FAIL_CONDITIONS) + '.xlsx')
    tf = preprocessData(tf)
    tf.to_excel('tfPreProcessed' + str(MAX_FAIL_CONDITIONS) + '.xlsx')

    return tf

# Function that shows what rules caused the patients to not receive the welfare benefit
def printFailOn(data, MAX_FAIL_CONDITIONS, DATA_POINTS):
    for i in range(len(data)):
        print(data[i], "of the patients failed on rule ", i+1, "out of the total", DATA_POINTS*2, "for ", MAX_FAIL_CONDITIONS, " maximum fail conditions")

# Function that generates noise data
# The paper mentions that there should be 52 noise variables, which are randomy values between 0 and 100
def generateNoiseData(DATA_POINTS):
    noiseVariables = []
    keys = []

    for i in range(NOISE_VARIABLES):
        df1 = generateNumerical(0, 100, DATA_POINTS)
        df2 = generateNumerical(0, 100, DATA_POINTS)
        name = "Noise" + str(i)
        keys.append(name)
        noiseVariables.append((df1+df2)) 

    df = pd.DataFrame.from_records(list(map(list, zip(*noiseVariables))), columns = keys)
    return df

# Preprocessing data such that the Residency, Spouse, InOut and Gender variables are all converted to numerical booleans
# 0 = No, 1 = Yes
# 0 = Out, 1 = In
# 0 = Female, 1 = Male
# This function can also easily be converted, so that the data already comes out as numerical in the initial data generation
def preprocessData(df):

    for i in range(len(df.Age)):

        if(df.loc[i, "Residency"] == "Yes"):
            df.loc[i, "Residency"] = 1
        if(df.loc[i, "Residency"] == "No"):
            df.loc[i, "Residency"] = 0

        if(df.loc[i, "Spouse"] == "Yes"):
            df.loc[i, "Spouse"] = 1
        if(df.loc[i, "Spouse"] == "No"):
            df.loc[i, "Spouse"] = 0

        if(df.loc[i, "InOut"] == "In"):
            df.loc[i, "InOut"] = 1
        if(df.loc[i, "InOut"] == "Out"):
            df.loc[i, "InOut"] = 0

        if(df.loc[i, "Gender"] == "Male"):
            df.loc[i, "Gender"] = 1
        if(df.loc[i, "Gender"] == "Female"):
            df.loc[i, "Gender"] = 0

    return df

def generateNumerical(min, max, DATA_POINTS):
    sample = list(range(min, max+1))
    data = []
    for i in range(DATA_POINTS):
        data.append(random.choice(sample))

    return data

def generateBoolean(valueList, DATA_POINTS):
    data = []
    for i in range(DATA_POINTS):
            data.append(random.choice(valueList))

    return data

# Function that generates a list of size DATA_POINTS
# All variables in the list are uniformly distributed
# Each datapoint in the list has size 12, including numerical values and boolean values
# All datapoints in the list are eligible
def generatePerfectData(DATA_POINTS):

    # Generating numerical data:
    ageData       = generateNumerical(60,100, DATA_POINTS)
    resourceData  = generateNumerical(0, 3000, DATA_POINTS)
    distData      = generateNumerical(0, 40, DATA_POINTS)

    # Generating numerical contribution year data
    contrData1    = generateNumerical(1, 1, DATA_POINTS)
    contrData2    = generateNumerical(1, 1, DATA_POINTS)
    contrData3    = generateNumerical(1, 1, DATA_POINTS)
    contrData4    = generateNumerical(1, 1, DATA_POINTS)
    contrData5    = generateNumerical(1, 1, DATA_POINTS)

    # Generating boolean data:
    # To avoid preprocessing, these should be changed to numerical
    residencyData = generateBoolean(["Yes"], DATA_POINTS)
    spouseData    = generateBoolean(["Yes"], DATA_POINTS)
    inoutData     = generateBoolean(["In", "Out"], DATA_POINTS)
    genderData    = generateBoolean(["Male", "Female"], DATA_POINTS)

    # Combining the dataset for a pandas dataframe
    dat = {'Age': ageData, 'Resource': resourceData, 'Distance': distData,
           'Residency': residencyData, 'Spouse': spouseData, 'InOut': inoutData, 'Gender': genderData,
           'Contribution1': contrData1, 'Contribution2': contrData2, 'Contribution3': contrData3, 'Contribution4': contrData4, 'Contribution5': contrData5}

    df = pd.DataFrame(data = dat)

    # Correcting potential issues in the otherwise perfect dataset
    for i in range(len(df.Age)):

        if(df.loc[i, "Gender"] == "Male" and df.loc[i, "Age"] < 65):
            df.loc[i, "Age"] = random.choice(list(range(65, 100)))
        
        if(df.loc[i, "InOut"] == "In" and df.loc[i, "Distance"] > IN_OUT_PATIENT_DISTANCE):
            df.loc[i, "InOut"] = "Out"

        if(df.loc[i, "InOut"] == "Out" and df.loc[i, "Distance"] < IN_OUT_PATIENT_DISTANCE):
            df.loc[i, "InOut"] = "In"

    return df

# This function checks the eligibility of each row in the dataframe
# It checks if each of the six rules are correct or not
def checkEligibility(df):
    eligibility = []
    eligibilityVal = 1
    failOn = [0,0,0,0,0,0]

    for i in range(len(df.Age)):
        totalContributionYears = 0
        if(((df.loc[i, "Gender"] == "Male") & (df.loc[i, "Age"] < 65)) | ((df.loc[i, "Gender"] == "Female") & (df.loc[i, "Age"] < 60))):
            eligibilityVal = 0
            failOn[0] += 1

        totalContributionYears = df.loc[i, "Contribution1"] + df.loc[i, "Contribution2"] + df.loc[i, "Contribution3"] + df.loc[i, "Contribution4"] + df.loc[i, "Contribution5"]
        if(totalContributionYears < 4):
            eligibilityVal = 0
            failOn[1] += 1

        if(df.loc[i, "Spouse"] == "No"):
            eligibilityVal = 0
            failOn[2] += 1

        if(df.loc[i, "Residency"] == "No"):
            eligibilityVal = 0
            failOn[3] += 1
        
        if(df.loc[i, "Resource"] > 3000):
            eligibilityVal = 0
            failOn[4] += 1

        if((df.loc[i, "InOut"] == "In" and df.loc[i, "Distance"] > IN_OUT_PATIENT_DISTANCE) or (df.loc[i, "InOut"] == "Out" and df.loc[i, "Distance"] < IN_OUT_PATIENT_DISTANCE)):
            eligibilityVal = 0
            failOn[5] += 1

        eligibility.append(eligibilityVal)
        eligibilityVal = 1

    df['Eligibility'] = eligibility

    return (df, failOn)

# Function that prints the eligibility percentage over the entire dataset
def calcEligibilityPerc(df):
    
    truePerc = 0
    for i in range(len(df.Eligibility)):
        if(df.loc[i, "Eligibility"] == 1):
            truePerc += 1
    
    truePerc /= len(df.Eligibility); truePerc *= 100
    
    print(truePerc, "% of the dataset is eligible for a welfare benefit for a visit")

# Function that modifies a (perfect) dataset, so that it fails on one or multiple conditions
# The six rules' failing conditions are specified here as well
def failConditions(df, MAX_FAIL_CONDITIONS):

    for i in range(len(df.Age)):
        # An amount of fails is generated for each row
        # The rules that it failed on are saved in conditions, so that it does not fail multiple times on the same rule
        NR_FAIL_CONDITIONS = random.choice(list(range(1, MAX_FAIL_CONDITIONS+1)))
        conditions = [0,0,0,0,0,0]

        # Iterating over the amount of conditions that the row should fail on
        for j in range(NR_FAIL_CONDITIONS):
            # When the data point is supposed to fail on only one condition, this if else statement makes sure that the rule it fails on is completely fairly distributed.
            if (MAX_FAIL_CONDITIONS == 1) or (MAX_FAIL_CONDITIONS == 6 and j == 1):
                rand = int(i/(float(len(df.Age)/6)))
            else:
                rand = random.choice(list(range(0, 6)))

            # If the rule was already failed on, then it should get another rule to fail on.
            while conditions[rand] == 1:
                rand = random.choice(list(range(0, 6)))
            conditions[rand] = 1

            if rand == 0:
                if df.loc[i, "Gender"] == "Male":
                    df.loc[i, "Age"] = random.choice(list(range(0, 65)))

                if df.loc[i, "Gender"] == "Female":
                    df.loc[i, "Age"] = random.choice(list(range(0, 60)))
                continue

            if rand == 1:
                contrYears = [0,0,0,0,0]
                yearsPaid = random.choice(list(range(0, 3)))

                for q in range(yearsPaid):
                    contrYears[random.choice(list(range(0, 5)))] = 1

                df.loc[i, "Contribution1"] = contrYears[0]
                df.loc[i, "Contribution2"] = contrYears[1]
                df.loc[i, "Contribution3"] = contrYears[2]
                df.loc[i, "Contribution4"] = contrYears[3]
                df.loc[i, "Contribution5"] = contrYears[4]
                continue

            if rand == 2:
                df.loc[i, "Spouse"] = "No"
                continue

            if rand == 3:
                df.loc[i, "Residency"] = "No"
                continue

            if rand == 4:
                df.loc[i, "Resource"] = random.choice(list(range(3001, 3500)))
                continue

            if rand == 5:
                if df.loc[i, "InOut"] == "Out":
                    df.loc[i, "Distance"] = random.choice(list(range(0, IN_OUT_PATIENT_DISTANCE-1)))
                    continue

                if df.loc[i, "InOut"] == "In":
                    df.loc[i, "Distance"] = random.choice(list(range(IN_OUT_PATIENT_DISTANCE+1, 40)))
                    continue

    return df
   
if __name__ == "__main__":
    main()