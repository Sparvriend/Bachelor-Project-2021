import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# Function that defines the testing/training dataset and splits those into independent and the dependent variable
# Then it scales the data with a standard scaler and then returns the split and scaled data
def manualScaleSplit(train, test):
    x_train = train.drop('Eligibility', axis = 1); x_test = test.drop('Eligibility', axis = 1)
    y_train = train['Eligibility']; y_test = test['Eligibility']

    stdScaler = StandardScaler()
    stdScaler.fit(x_train)
    StandardScaler(copy = True, with_mean = True, with_std = True)
    x_train = stdScaler.transform(x_train)
    x_test = stdScaler.transform(x_test)

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

if __name__ == "__main__":
    main()