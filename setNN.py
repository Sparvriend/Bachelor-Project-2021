import pandas as pd
import keras
import dataGen

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Neural network setup Python file made for the Bachelor Project by Timo Wahl (s3812030)

# Neural network: 1/2/3 hidden layers (or more) with various shapes (paper used triangular) using back propogation
# Neural networks from paper:
# One hidden layer: 64 node input + 1 hidden layer with 12 nodes + 1 output layer with 1 node
# Two hidden layer: 64 node input + 1 hidden layer with 24 nodes + 1 hidden layer with 6 nodes + 1 ouput layer with 1 node
# Three hidden layer: 64 node input + 1 hidden layer with 24 nodes + 1 hidden layer with 10 nodes + 1 hidden layer with 3 nodes + 1 output layer with 1 node

# TODO:
# - Play around with parameters to get the same accuracies as in the paper
# - Accuracy decrease from single fail to multiple fail dissapeared :(
# - Ensure that data generation is all correct
# - Include two extra datasets (for age and distance) like in the paper (2 training datasets)
# - Add the printFailOn to the report for a nice overview in the results section
# - Brainstorm ideas for own research
 
def main():
    print("Generating training/testing data for the neural network")
    singleFail = dataGen.generateFailData(1, 2200)
    multipleFail = dataGen.generateFailData(6, 2200)
    
    print("Testing with single fail")
    testDataset(singleFail)
    print("Testing with multiple fail")
    testDataset(multipleFail)

# Function that calls the underlying functions for running the NN
def testDataset(df):
    x_train, x_test, y_train, y_test = scaleSplitData(df)
    models = getMLPModels()
    fitModels(models, x_train, x_test, y_train, y_test)

# Function that first splits the data into four equal lengths
# Then it scales the data with a standard scaler
# Then it returns the split and scaled data
def scaleSplitData(df):
    x = df.drop('Eligibility', axis = 1)
    y = df['Eligibility']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2000/4400)

    stdScaler = StandardScaler()
    stdScaler.fit(x_train)
    StandardScaler(copy = True, with_mean = True, with_std = True)
    x_train = stdScaler.transform(x_train)
    x_test = stdScaler.transform(x_test)

    return (x_train, x_test, y_train, y_test)

# Function that creates the three classifiers as mentioned in the paper
# The classifiers have a lot of parameters that can be played around with
def getMLPModels():
    models = []

    models.append(MLPClassifier(hidden_layer_sizes=(12), max_iter = 2000, activation = 'logistic'))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 6), max_iter = 2000, activation = 'logistic'))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), max_iter = 2000, activation = 'logistic'))

    return models

# Function that fits the models to the test data
# It also prints a simple statistical analysis of the data as well as a confusion matrix
def fitModels(models, x_train, x_test, y_train, y_test):

    for model in models:
        model.fit(x_train, y_train)

        predict = model.predict(x_test)

        print(model)
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predict))
        print("Classification report:")
        print(classification_report(y_test, predict))

if __name__ == "__main__":
    main()