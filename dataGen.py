import pandas as pd
import numpy as np
import random

def main():
    initData(1, 500)

# This function initializes data generation
# It is used for the normal dataset, 50% of the dataset is eligible, the other 50% is ineligible based on the number of fail conditions
def initData(MAX_FAIL_CONDITIONS, DATA_POINTS):
    dfFail = failConditions(generatePerfectData(DATA_POINTS), MAX_FAIL_CONDITIONS)
    tf = pd.concat([generatePerfectData(DATA_POINTS), dfFail[0]], axis = 0, ignore_index=True)
    tf = modifyData(tf, DATA_POINTS, dfFail[1])

    return tf

# This function modifies the data, it adds the eligibility column as well as the noise variables
# Followed by preprocessing
def modifyData(tf, DATA_POINTS, failOn):
    #printFailOn(failOn, tf, DATA_POINTS)
    #tf = pd.concat([tf, generateNoiseData(DATA_POINTS)], axis = 1)
    tf.to_excel('testing.xlsx')

    return tf

# Function that shows what rules caused the patients to not receive the welfare benefit
# Can be used for any dataset
def printFailOn(data, df, DATA_POINTS):
    for i in range(len(data)):
        print(data[i], "of the visitors failed on rule", i+1, "out of the total", DATA_POINTS*2)
    print("Percentage of visits allowed in this dataset: " + str(((df['allowed'].sum() / (DATA_POINTS*2))*100)) + " %.")

# Generating numerical data, based on a list of values that can be chosen from, the step variable is optional
def generateNumerical(min, max, DATA_POINTS, step = 1):
    sample = list(range(min, max+1, step))
    data = []
    for i in range(DATA_POINTS):
        data.append(random.choice(sample))

    return data

# Generating boolean data, based on a value list
def generateBoolean(DATA_POINTS):
    valueList = [0,1]
    data = []
    for i in range(DATA_POINTS):
            data.append(random.choice(valueList))

    return data

# Function that generates a list of size DATA_POINTS
# All datapoints in the list are eligible
def generatePerfectData(DATA_POINTS):
    # Generating numerical data
    lastHostVisit    = generateNumerical(24, 48, DATA_POINTS)
    lastVisitorVisit = generateNumerical(24, 48, DATA_POINTS)
    partySizeHost    = generateNumerical(1, 6, DATA_POINTS)
    partySizeVisitor = generateNumerical(1, 6, DATA_POINTS)
    adultsVisitor    = np.ones(DATA_POINTS)

    # Generating boolean data
    corTestHost     = np.zeros(DATA_POINTS); corTestVisitor     = np.zeros(DATA_POINTS)
    quarHost        = np.zeros(DATA_POINTS); quarVisitor        = np.zeros(DATA_POINTS)
    dryCoughHost    = np.zeros(DATA_POINTS); dryCoughVisitor    = np.zeros(DATA_POINTS)
    feverHost       = np.zeros(DATA_POINTS); feverVisitor       = np.zeros(DATA_POINTS)
    headacheHost    = np.zeros(DATA_POINTS); headacheVisitor    = np.zeros(DATA_POINTS)
    shortBreathHost = np.zeros(DATA_POINTS); shortBreathVisitor = np.zeros(DATA_POINTS)
    soreThroatHost  = np.zeros(DATA_POINTS); soreThroatVisitor  = np.zeros(DATA_POINTS)
    hostInNed       =  np.ones(DATA_POINTS); visitAllowed       = np.ones(DATA_POINTS)

    # Creating the host and visitor pandas dataframes and then concatenating them
    hostDF = pd.DataFrame({'lastVisitH':lastHostVisit, 'sizeH':partySizeHost, 'corTestH':corTestHost, 'quarH':quarHost, 'symp1H':dryCoughHost, 
               'symp2H':feverHost, 'symp3H':headacheHost, 'symp4H':shortBreathHost, 'symp5H':soreThroatHost, 'inNedH':hostInNed})
    visitorDF = pd.DataFrame({'lastVisitV':lastVisitorVisit, 'sizeV':partySizeVisitor, 'corTestV':corTestVisitor, 'quarV':quarVisitor, 'symp1V':dryCoughVisitor,
     'symp2V':feverVisitor, 'symp3V':headacheVisitor, 'symp4V':shortBreathVisitor, 'symp5V':soreThroatVisitor, 'adults':adultsVisitor})
    df = pd.concat([hostDF, visitorDF], axis = 1)

    # Ensuring that the dataset is perfect
    for i in range(len(df.quarH)):
        if(df.loc[i, 'sizeH'] + df.loc[i, 'sizeV'] > 6):
            randy = random.choice(list(range(0, 2)))
            if randy == 0:
                df.loc[i, 'sizeH'] = random.choice(list(range(1, 6)))
                df.loc[i, 'sizeV'] = random.choice(list(range(1, 7-df.loc[i, 'sizeH'])))
            if randy == 1:
                df.loc[i, 'sizeV'] = random.choice(list(range(1, 6)))
                df.loc[i, 'sizeH'] = random.choice(list(range(1, 7-df.loc[i, 'sizeV'])))
    
    # Adding that each of these datapoints is allowed to visit
    df = pd.concat([df, pd.DataFrame({'allowed': visitAllowed})], axis = 1)

    return df

# Function that modifies a (perfect) dataset, so that it fails on one or multiple conditions
# The eight rules' failing conditions are specified here as well
def failConditions(df, MAX_FAIL_CONDITIONS):
    failOn = np.zeros(8)
    for i in range(len(df.quarH)):
        df.loc[i, 'allowed'] = 0
        # An amount of fails is generated for each row
        # The rules that it failed on are saved in conditions, so that it does not fail multiple times on the same rule
        NR_FAIL_CONDITIONS = random.choice(list(range(1, MAX_FAIL_CONDITIONS+1)))
        conditions = np.zeros(8)

        # Iterating over the amount of conditions that the row should fail on
        for j in range(NR_FAIL_CONDITIONS):
            # When the data point is supposed to fail on only one condition, this if else statement makes sure that the rule it fails on is completely fairly distributed.
            if (MAX_FAIL_CONDITIONS == 1) or (MAX_FAIL_CONDITIONS == 8 and j == 1):
                rand = int(i/(float(len(df.quarH)/8)))
            else:
                rand = random.choice(list(range(0, 8)))

            # If the rule was already failed on, then it should get another rule to fail on.
            while conditions[rand] == 1:
                rand = random.choice(list(range(0, 8)))
            conditions[rand] = 1

            if rand == 0:
                df.loc[i, 'lastVisitH'] = random.choice(list(range(0, 24)))
                failOn[0] += 1
                continue

            if rand == 1:
                df.loc[i, 'lastVisitV'] = random.choice(list(range(0, 24)))
                failOn[1] += 1
                continue

            if rand == 2:              
                if df.loc[i, 'sizeV'] == 1:
                    df.loc[i, 'sizeV'] = random.choice(list(range(2, 6)))
                
                if df.loc[i, 'sizeV'] == 2:
                    df.loc[i, 'adults'] = 2
                
                if df.loc[i, 'sizeV'] > 2:
                    df.loc[i, 'adults'] = random.choice(list(range(2, df.loc[i, 'sizeV'])))

                failOn[2] += 1
                continue

            if rand == 3:
                # 0 = only the visitor has a corona test, 1 = only the host has a corona test, 2 = both have a corona test
                randy = random.choice(list(range(0, 3)))
                if randy == 0:
                    df.loc[i, 'corTestH'] = 1
                if randy == 1:
                    df.loc[i, 'corTestV'] = 1
                if randy == 2:
                    df.loc[i, 'corTestH'] = 1; df.loc[i, 'corTestV'] = 1
                failOn[3] += 1
                continue

            if rand == 4:
                # 0 = only the visitor is in quarantine, 1 = only the host is in quarantine, 2 = both are in quarantine
                randy = random.choice(list(range(0, 3)))
                if randy == 0:
                    df.loc[i, 'quarH'] = 1
                if randy == 1:
                    df.loc[i, 'quarV'] = 1
                if randy == 2:
                    df.loc[i, 'quarH'] = 1; df.loc[i, 'quarV'] = 1
                failOn[4] += 1
                continue

            if rand == 5:
                hostSymptomps = np.zeros(5); hostSymptomCount = random.choice(list(range(1, 6)))
                visitorSymptomps = np.zeros(5); visitorSymptomCount = random.choice(list(range(1, 6)))
                
                for q in range(hostSymptomCount):
                    hostSymptomps[random.choice(list(range(0, 5)))] = 1
                for q in range(visitorSymptomCount):
                    visitorSymptomps[random.choice(list(range(0, 5)))] = 1
                for q in range(len(hostSymptomps)):
                    df.loc[i, "symp" + str(q+1) + "H"] = hostSymptomps[q]
                for q in range(len(visitorSymptomps)):
                    df.loc[i, "symp" + str(q+1) + "V"] = visitorSymptomps[q]
                failOn[5] += 1
                continue
            
            if rand == 6:
                df.loc[i, 'inNedH'] = 0
                failOn[6] += 1
                continue
            
            if rand == 7:
                # It is assumed that if the party size increases, the amount of kids (age < 17) is increased while the amount of adults stays the same
                randy = random.choice(list(range(0, 2)))
                if randy == 0:
                    df.loc[i, 'sizeH'] = random.choice(list(range(1, 6)))
                    df.loc[i, 'sizeV'] = random.choice(list(range(7 - df.loc[i, 'sizeH'], 7)))
                if randy == 1:
                    df.loc[i, 'sizeV'] = random.choice(list(range(1, 6)))
                    df.loc[i, 'sizeH'] = random.choice(list(range(7 - df.loc[i, 'sizeV'], 7)))
                failOn[7] += 1
                continue

    return (df, failOn)

if __name__ == "__main__":
    main()