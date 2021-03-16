import pandas as pd
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# Neural network setup Python file made for the Bachelor Project by Timo Wahl (s3812030)

# Neural network: 1/2/3 hidden layers (or more) with various shapes (paper used triangular) using back propogation
# Neural networks from paper:
# One hidden layer: 64 node input + 1 hidden layer with 12 nodes + 1 output layer with 1 node
# Two hidden layer: 64 node input + 1 hidden layer with 24 nodes + 1 hidden layer with 6 nodes + 1 ouput layer with 1 node
# Three hidden layer: 64 node input + 1 hidden layer with 24 nodes + 1 hidden layer with 10 nodes + 1 hidden layer with 3 nodes + 1 output layer with 1 node

# TODO:
# - Play around with parameters to get the same accuracies as in the paper
# - Include two extra datasets (for age and distance) like in the paper (2 training datasets)
# - Add the printFailOn to the report for a nice overview in the results section
# - Brainstorm ideas for own research
 
# Function that calls the underlying functions for running the NN
def testDataset(train, test):
    #x_train, x_test, y_train, y_test = scaleSplitData(df)
    x_train, x_test, y_train, y_test = manualScaleSplit(train, test)
    models = getMLPModels()
    predictions = fitModels(models, x_train, x_test, y_train, y_test)
    print("-------------------------------------------------------------------------------------------------------------")

    # Returning the prediction for the age and distance datasets
    return predictions

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

def manualScaleSplit(train, test):
    x_train = train.drop('Eligibility', axis = 1)
    x_test = test.drop('Eligibility', axis = 1)
    y_train = train['Eligibility']
    y_test = test['Eligibility']

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

    # Data is already shuffled
    models.append(MLPClassifier(hidden_layer_sizes=(12), max_iter = 5000, activation = 'logistic', shuffle = False))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 6), max_iter = 5000, activation = 'logistic', shuffle = False))
    models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), max_iter = 5000, activation = 'logistic', shuffle = False))

    return models

# Function that fits the models to the test data
# It also prints a simple statistical analysis of the data as well as a confusion matrix
def fitModels(models, x_train, x_test, y_train, y_test):
    predicts = []

    for model in models:
        model.fit(x_train, y_train)

        predict = model.predict(x_test)

        print(model)
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predict))
        print("accuracy score:")
        print(accuracy_score(y_test, predict))

        # print("Classification report:")
        # print(classification_report(y_test, predict))

        predicts.append(predict)

    # fix that you actually return the three different predicts not just the last one
    return predicts

if __name__ == "__main__":
    main()