import pandas as pd
import random

# Constant for the in out patient distance, measured in km
# This is copied directly from the paper
IN_OUT_PATIENT_DISTANCE = 50
NOISE_VARIABLES = 52
    
# Data generation Python file made for the Bachelor Project by Timo Wahl (s3812030)
# The paper that is constantly mentioned in this file and relates to the other replication files:
# https://dl.acm.org/doi/abs/10.1145/158976.159012?casa_token=cTqiK-PMwnEAAAAA:KtSh_D8f5J3cV4sqSH3qyKG-XhHAb28hNt0au51BNDl4VdSQQ6aKp1W_baNu2aJ6O7LPL1YbOhhX

# This function initializes data generation
# It is used for the normal dataset, 50% of the dataset is eligible, the other 50% is ineligible based on the number of fail conditions
def initData(MAX_FAIL_CONDITIONS, DATA_POINTS, TYPE, LOC):
    dfFail = failConditions(generatePerfectData(DATA_POINTS), MAX_FAIL_CONDITIONS)
    tf = pd.concat([generatePerfectData(DATA_POINTS), dfFail], axis = 0, ignore_index=True)
    tf = modifyData(tf, DATA_POINTS, TYPE, LOC)

    return tf

# This function modifies the data, it adds the eligibility column as well as the noise variables
# Followed by preprocessing
def modifyData(tf, DATA_POINTS, TYPE, LOC):
    eligibilityResult = checkEligibility(tf)
    tf = eligibilityResult[0]

    # calcEligibilityPerc(tf)
    # printFailOn(eligibilityResult[1], DATA_POINTS)

    tf = pd.concat([tf, generateNoiseData(DATA_POINTS)], axis = 1)
    tf = preprocessData(tf)
    tf.to_excel(str(LOC) + str(TYPE) + '.xlsx')

    return tf

# Function that shows what rules caused the patients to not receive the welfare benefit
# Can be used for any dataset
def printFailOn(data, DATA_POINTS):
    for i in range(len(data)):
        print(data[i], "of the patients failed on rule", i+1, "out of the total", DATA_POINTS*2)

# Function that generates noise data
# The paper mentions that there should be 52 noise variables, which are random values between 0 and 100
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

# Generating numerical data, based on a list of values that can be chosen from, the step variable is optional
def generateNumerical(min, max, DATA_POINTS, step = 1):
    sample = list(range(min, max+1, step))
    data = []
    for i in range(DATA_POINTS):
        data.append(random.choice(sample))

    return data

# Generating boolean data, based on a value list
def generateBoolean(valueList, DATA_POINTS):
    data = []
    for i in range(DATA_POINTS):
            data.append(random.choice(valueList))

    return data

# Function that generates a list of size DATA_POINTS
# All datapoints in the list are eligible
def generatePerfectData(DATA_POINTS):

    # Generating numerical data
    ageData       = generateNumerical(60,100, DATA_POINTS, 5)
    resourceData  = generateNumerical(0, 3000, DATA_POINTS)
    distData      = generateNumerical(0, 100, DATA_POINTS)

    # Generating numerical contribution year data
    contrData1    = generateNumerical(1, 1, DATA_POINTS)
    contrData2    = generateNumerical(1, 1, DATA_POINTS)
    contrData3    = generateNumerical(1, 1, DATA_POINTS)
    contrData4    = generateNumerical(1, 1, DATA_POINTS)
    contrData5    = generateNumerical(1, 1, DATA_POINTS)

    # Generating boolean data
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

    # Ensuring that the dataset is perfect
    for i in range(len(df.Age)):
        if(df.loc[i, "Gender"] == "Male" and df.loc[i, "Age"] < 65):
            df.loc[i, "Age"] = random.choice(list(range(65, 100, 5)))
        
        if(df.loc[i, "InOut"] == "In" and df.loc[i, "Distance"] > IN_OUT_PATIENT_DISTANCE):
            df.loc[i, "InOut"] = "Out"

        if(df.loc[i, "InOut"] == "Out" and df.loc[i, "Distance"] < IN_OUT_PATIENT_DISTANCE):
            df.loc[i, "InOut"] = "In"

    return df

# This function checks the eligibility of each row in the dataframe
# It checks if each of the six rules are correct or not
# It returns the same dataframe with an added column for eligibility
# It also returns a list which shows how many cases failed on which rules
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
                    df.loc[i, "Age"] = random.choice(list(range(0, 65, 5)))

                if df.loc[i, "Gender"] == "Female":
                    df.loc[i, "Age"] = random.choice(list(range(0, 60, 5)))
                continue

            if rand == 1:
                contrYears = [0,0,0,0,0]
                yearsPaid = random.choice(list(range(0, 3)))

                for q in range(yearsPaid):
                    contrYears[random.choice(list(range(0, 5)))] = 1
                for q in range(len(contrYears)):
                    df.loc[i, "Contribution" + str(q+1)] = contrYears[q]
                continue

            if rand == 2:
                df.loc[i, "Spouse"] = "No"
                continue

            if rand == 3:
                df.loc[i, "Residency"] = "No"
                continue

            if rand == 4:
                df.loc[i, "Resource"] = random.choice(list(range(3001, 6000)))
                continue

            if rand == 5:
                if df.loc[i, "InOut"] == "Out":
                    df.loc[i, "Distance"] = random.choice(list(range(0, IN_OUT_PATIENT_DISTANCE-1)))
                    continue

                if df.loc[i, "InOut"] == "In":
                    df.loc[i, "Distance"] = random.choice(list(range(IN_OUT_PATIENT_DISTANCE+1, 100)))
                    continue

    return df