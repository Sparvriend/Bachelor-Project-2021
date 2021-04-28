import classifier
import normalDataset
import distanceDataset
import ageDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

DATA_POINTS = 5000
MAX_ITERATIONS = 3000

def main():
    print("0 = MLP, 1 = Random Forest, 2 = XGBoost")
    ls = input("Which learning system?\n"); ls = int(ls)
    print("0 = Normal, 1 = Distance, 2 = Age")
    ds = input("Which dataset?\n"); ds = int(ds)
    if ls > 3 or 0 > ls or ds > 2 or 0 > ds:
        print("Error input value")
        exit()
    learningSystems = ["MLP", "Random_Forest", "XGBoost"] 
    if ls < 2 or 0 < ls: 
        models = []
        if ls == 0:
            print("Running MLP learning system")
            models = getMLPModels(MAX_ITERATIONS)
        if ls == 1:
            print("Running Random Forest learning system")
            models = getRandomForestModels()
        if ls == 2:
            print("Running XGBoost learning system")
            models = getXGBoostModels()
        dat = []
        if ds == 0:
            print("On normal dataset")
            dat = normalDataset.getData(DATA_POINTS)
            runClassifierNormal(dat, models)
        if ds == 1:
            print("On distance dataset")
            dat = distanceDataset.getData(DATA_POINTS)
            out = runClassifierDistance(dat, models)
            distanceDataset.printGraph(dat[2], out[0], 'SFDistance' + str(learningSystems[ls]))
            distanceDataset.printGraph(dat[2], out[1], 'MFDistance' + str(learningSystems[ls]))
        if ds == 2:
            print("On age dataset")
            dat = ageDataset.getData(DATA_POINTS)
            out = runClassifierAge(dat, models)
            ageDataset.printGraph(dat[2], out[0], 'SFAge' + str(learningSystems[ls]))
            ageDataset.printGraph(dat[2], out[1], 'MFAge' + str(learningSystems[ls]))

# Function that creates the three MLP classifiers
def getMLPModels(iterations):
    models = []
    models.append(MLPClassifier(hidden_layer_sizes=(12), max_iter = iterations, activation = 'logistic', learning_rate_init = 0.001, batch_size = 50))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 6), max_iter = iterations, activation = 'logistic', learning_rate_init = 0.001, batch_size = 50))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), max_iter = iterations, activation = 'logistic', learning_rate_init = 0.001, batch_size = 50))

    return models

# Function that creates Random Forest classifiers
def getRandomForestModels():
    models = []
    models.append(RandomForestClassifier(n_estimators = 10, random_state = 0))
    models.append(RandomForestClassifier(n_estimators = 20, random_state = 0))
    models.append(RandomForestClassifier(n_estimators = 50, random_state = 0))

    return models

# Function that creates XGBoost classifiers
def getXGBoostModels():
    models = []
    models.append(XGBClassifier(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 5, n_estimators = 10, verbosity = 0))
    models.append(XGBClassifier(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 5, n_estimators = 20, verbosity = 0))
    models.append(XGBClassifier(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 5, n_estimators = 50, verbosity = 0))

    return models

# Function that runs the normal dataset
def runClassifierNormal(datasets, models):
    print("Running the learning system on the normal dataset")
    print("Training on multiple fail, testing on multiple fail"); acc1 = classifier.testDataset(datasets[2], datasets[3], models)[1]
    print("Training on multiple fail, testing on single fail"); acc2 = classifier.testDataset(datasets[2], datasets[1], models)[1]
    print("Training on single fail, testing on multiple fail"); acc3 = classifier.testDataset(datasets[0], datasets[3], models)[1]
    print("Training on single fail, testing on single fail"); acc4 = classifier.testDataset(datasets[0], datasets[1], models)[1]

    return (acc1 + acc2 + acc3 + acc4)

def runClassifierDistance(datasets, models):
    print("Running the learning system on the distance dataset")
    print("Training on single fail, testing on distance"); pred1 = classifier.testDataset(datasets[0], datasets[2], models)[0]
    print("Training on multiple fail, testing on distance"); pred2 = classifier.testDataset(datasets[1], datasets[2], models)[0]

    return (pred1, pred2)

def runClassifierAge(datasets, models):
    print("Running the learning system on the age dataset")
    print("Training on single fail, testing on age"); pred1 = classifier.testDataset(datasets[0], datasets[2], models)[0]
    print("Training on multiple fail, testing on age"); pred2 = classifier.testDataset(datasets[1], datasets[2], models)[0]

    return (pred1, pred2)

if __name__ == "__main__":
    main()