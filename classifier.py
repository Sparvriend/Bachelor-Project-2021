from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np

# This file is used to create, run and scale the different classifiers used in this project.
# The classifiers can then be used to be either trained or tested on.

# Function that receives the machine learning models based on the input value.
# There are two of every one, one for single fail training and one for multiple fail training.
def getModels(ls):
    models = []
    if ls == 0:
        print("Getting MLP models")
        models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), activation = 'logistic', alpha = 0.00008, learning_rate_init = 0.008, batch_size = 26, max_iter = 3000))
        models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), activation = 'logistic', alpha = 0.00008, learning_rate_init = 0.008, batch_size = 26, max_iter = 3000))
    if ls == 1:
        print("Getting Random Forest models")
        models.append(RandomForestClassifier(n_estimators = 16, max_depth = 19, max_leaf_nodes = 17, min_samples_split = 6, random_state = 0))
        models.append(RandomForestClassifier(n_estimators = 16, max_depth = 19, max_leaf_nodes = 17, min_samples_split = 6, random_state = 0))
    if ls == 2:
        print("Getting XGBoost models")
        models.append(XGBClassifier(n_estimators = 16, max_depth = 7, objective ='reg:squarederror', learning_rate = 0.25, gamma = 0.5, verbosity = 0))
        models.append(XGBClassifier(n_estimators = 16, max_depth = 7, objective ='reg:squarederror', learning_rate = 0.25, gamma = 0.5, verbosity = 0))
    return models

# This function scales the data with a minMax scaler.
# It scales some variables manually, because there is a chance that the max value is not present in the list
# causing the minMaxScaler to not scale it correctly.
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

# Function that retrieves two minMaxScalers
def getScalers(singleFail, multipleFail):
    scalers = []; scalers.append(getScaler(singleFail)); scalers.append(getScaler(multipleFail))
    return scalers

def getScaler(data):
    x_train = data.drop(['Age', 'Resource', 'Distance', 'Eligible'], axis = 1)
    scaler = MinMaxScaler(); scaler.fit(x_train)
    return scaler

# Function that tests the model on a dataset, it returns the predictions and the accuracies.
def onlyTest(trainedModel, testSet, scaler):
    x_test, y_test = scaleData(testSet, scaler)
    predict = trainedModel.predict(x_test)
    accuracy = accuracy_score(y_test, predict)
    return (predict, accuracy)

# Function that trains a model on a dataset.
def trainModel(trainSet, model, scaler):
    x_train, y_train = scaleData(trainSet, scaler)
    return model.fit(x_train, y_train)