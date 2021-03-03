import pandas as pd
import keras
import dataGen

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Neural network setup Python file made for the Bachelor Project by Timo Wahl (s3812030)

# TODO:
# - Neural network: 1/2/3 hidden layers (or more) with various shapes (paper used triangular) using back propogation
# - Neural networks from paper:
# One hidden layer: 64 node input + 1 hidden layer with 12 nodes + 1 output layer with 1 node
# Two hidden layer: 64 node input + 1 hidden layer with 24 nodes + 1 hidden layer with 6 nodes + 1 ouput layer with 1 node
# Three hidden layer: 64 node input + 1 hidden layer with 24 nodes + 1 hidden layer with 10 nodes + 1 hidden layer with 3 nodes + 1 output layer with 1 node
# - In the model it is mentioned that the NN is run on 2000 testcases, what does that mean exactly? Was the NN not split in 4 for the train and test data?

def main():
    testDataset(dataGen.generateFailData(1))
    testDataset(dataGen.generateFailData(6))

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
    x_train, x_test, y_train, y_test = train_test_split(x, y)

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

    models.append(MLPClassifier(hidden_layer_sizes=(12), max_iter = 3000))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 6), max_iter = 3000))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), max_iter = 3000))

    return models

# Function that fits the models to the test data
# It also prints a simple statistical analysis of the data as well as a confusion matrix
def fitModels(models, x_train, x_test, y_train, y_test):

    for model in models:
        model.fit(x_train, y_train)

        predict = model.predict(x_test)

        print(model)
        print(confusion_matrix(y_test, predict))
        print(classification_report(y_test, predict))

if __name__ == "__main__":
    main()