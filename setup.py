import classifier
import normalDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

DATA_POINTS = 1000
MAX_ITERATIONS = 3000

# TODO:
# - Include switch case for selecting the dataset to test on

def main():
    print("0 = MLP, 1 = randomForest, 2 = XGBoost")
    value = input("Which learning system?\n"); value = int(value)
    if value > 1 or 0 > value:
        print("Error value")
        exit()
    dat = normalDataset.getData(DATA_POINTS)
    models = []
    if value == 0:
        print("Running MLP learning system")
        models = getMLPModels(MAX_ITERATIONS)
    if value == 1:
        print("Running Random Forest learning system")
        models = getRandomForestModels()

    runClassifier(dat, models)

# Function that creates the three MLP classifiers
def getMLPModels(iterations):
    models = []
    models.append(MLPClassifier(hidden_layer_sizes=(12), max_iter = iterations, activation = 'relu', learning_rate_init = 0.001, batch_size = 50))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 6), max_iter = iterations, activation = 'relu', learning_rate_init = 0.001, batch_size = 50))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), max_iter = iterations, activation = 'relu', learning_rate_init = 0.001, batch_size = 50))

    return models

# Function that creates Random Forest Classifiers
def getRandomForestModels():
    models = []
    models.append(RandomForestClassifier(n_estimators = 10, random_state = 0))
    models.append(RandomForestClassifier(n_estimators = 20, random_state = 0))
    models.append(RandomForestClassifier(n_estimators = 50, random_state = 0))

    return models

def runClassifier(datasets, models):
    print("Running the learning system on the normal dataset")
    print("Training on multiple fail, testing on multiple fail"); acc1 = classifier.testDataset(datasets[2], datasets[3], models)[1]
    print("Training on multiple fail, testing on single fail"); acc2 = classifier.testDataset(datasets[2], datasets[1], models)[1]
    print("Training on single fail, testing on multiple fail"); acc3 = classifier.testDataset(datasets[0], datasets[3], models)[1]
    print("Training on single fail, testing on single fail"); acc4 = classifier.testDataset(datasets[0], datasets[1], models)[1]

if __name__ == "__main__":
    main()