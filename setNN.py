import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Neural network setup Python file made for the Bachelor Project by Timo Wahl (s3812030)

# Neural network: 1/2/3 hidden layers (or more) with various shapes (paper used triangular) using back propogation
# Directly copied from the paper, as mentioned in dataGen
# Setting the iterations to higher values does not result in higher accuracies
# Changing the solver/learning rate/activation does not improve the accuracy w.r.t the paper accuracies
# Increasing test/train dataset sizes increases accuracies, but diminishes the results from the paper
# For example result 2 has much too high accuracies when increasing the dataset sizes
 
# Function that calls the underlying functions for running the NN
# test/training datasets are manually split to have the right test/train sets
def testDataset(train, test, iterations):
    x_train, x_test, y_train, y_test = manualScaleSplit(train, test)
    models = getMLPModels(iterations)
    predictions, accuracies = fitModels(models, x_train, x_test, y_train, y_test)
    print("-------------------------------------------------------------------------------------------------------------")

    # Returning the prediction, used for the age and distance datasets
    return (predictions, accuracies)

def scaleData(data, scaler):
    x_train = data.drop('Eligible', axis = 1); y_train = data['Eligible']
    x_train_manual = x_train[['Age', 'Resource', 'Distance']]
    x_train = x_train.drop(['Age', 'Resource', 'Distance'], axis = 1)
    x_train = scaler.transform(x_train)

    for i in range(len(x_train_manual.Age)):
        x_train_manual.loc[i, 'Age'] /= 100
        x_train_manual.loc[i, 'Distance'] /= 100
        x_train_manual.loc[i, 'Resource'] /= 10000
    return (np.concatenate([x_train_manual, x_train], axis = 1), y_train)

def getScaler(data):
    x_train = data.drop(['Age', 'Resource', 'Distance', 'Eligible'], axis = 1)
    scaler = MinMaxScaler(); scaler.fit(x_train)
    return scaler

def manualScaleSplit(train, test):
    scaler = getScaler(train)
    x_train, y_train = scaleData(train, scaler)
    x_test, y_test = scaleData(test, scaler)

    return (x_train, x_test, y_train, y_test)

# Function that creates the three MLP classifiers as mentioned in the paper
# The activation is set to logistic as relu did not have traction yet in 1993, other parameters are set based on optimality
def getMLPModels(iterations):
    models = []

    models.append(MLPClassifier(hidden_layer_sizes=(12), max_iter = iterations, activation = 'logistic', learning_rate_init = 0.001, batch_size = 50))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 6), max_iter = iterations, activation = 'logistic', learning_rate_init = 0.001, batch_size = 50))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), max_iter = iterations, activation = 'logistic', learning_rate_init = 0.001, batch_size = 50))

    return models

# Function that fits the models to the test data
# It prints analysis data, as well as returning that data
def fitModels(models, x_train, x_test, y_train, y_test):
    predicts = []
    accuracies = []

    for model in models:
        model.fit(x_train, y_train)
        predict = model.predict(x_test)
        accuracy = accuracy_score(y_test, predict)
        print(model)
        print("Accuracy score:\n" + str(accuracy))
        predicts.append(predict)
        accuracies.append(accuracy)

    return (predicts, accuracies)
