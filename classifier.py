from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import setup

def onlyTest(trainedModel, testSet, scaler):
    x_test = testSet.drop('Eligible', axis = 1); y_test = testSet['Eligible']

    x_test_copy = x_test.copy()
    x_test_manual = x_test[['Age', 'Resource', 'Distance']]
    x_test = x_test.drop(['Age', 'Resource', 'Distance'], axis = 1)
    x_test = scaler.transform(x_test)

    for i in range(len(x_test_manual.Age)):
        x_test_manual.loc[i, 'Age'] /= 100
        x_test_manual.loc[i, 'Distance'] /= 100
        x_test_manual.loc[i, 'Resource'] /= 10000
    x_test = np.concatenate([x_test_manual, x_test], axis = 1)

    predict = trainedModel.predict(x_test)
    accuracy = accuracy_score(y_test, predict)
    print("Accuracy score: " + str(accuracy))
    #print("Confusion matrix: \n" + str(confusion_matrix(y_test, predict)))
    #pd.DataFrame(x_test, columns = x_test_copy.columns).to_excel('DataRes/testedOn.xlsx')

    return (predict, accuracy)

def trainModel(trainSet, model, scaler, type):
    x_train = trainSet.drop('Eligible', axis = 1); y_train = trainSet['Eligible']

    x_train_copy = x_train.copy()
    x_train_manual = x_train[['Age', 'Resource', 'Distance']]
    x_train = x_train.drop(['Age', 'Resource', 'Distance'], axis = 1)
    x_train = scaler.transform(x_train)

    for i in range(len(x_train_manual.Age)):
        x_train_manual.loc[i, 'Age'] /= 100
        x_train_manual.loc[i, 'Distance'] /= 100
        x_train_manual.loc[i, 'Resource'] /= 10000

    x_train = np.concatenate([x_train_manual, x_train], axis = 1)

    return model.fit(x_train, y_train)

# Function that calls the underlying functions for running the classifier
# Test/training datasets are manually split to have the right test/train sets
def testDataset(train, test, models):
    x_train, x_test, y_train, y_test = manualScaleSplit(train, test)
    predictions, accuracies = fitModels(models, x_train, x_test, y_train, y_test)
    print("-------------------------------------------------------------------------------------------------------------")

    # Returning the prediction, used for the age and distance datasets
    return (predictions, accuracies)

# Function that defines the testing/training dataset and splits those into independent and the dependent variable
# Then it scales the data with a standard scaler and then returns the split and scaled data
def manualScaleSplit(train, test):
    x_train = train.drop('Eligible', axis = 1); x_test = test.drop('Eligible', axis = 1)
    y_train = train['Eligible']; y_test = test['Eligible']

    x_train_copy = x_train.copy(); x_test_copy = x_test.copy()

    x_train_manual = x_train[['Age', 'Resource', 'Distance']]; x_train = x_train.drop(['Age', 'Resource', 'Distance'], axis = 1)
    x_test_manual = x_test[['Age', 'Resource', 'Distance']]; x_test = x_test.drop(['Age', 'Resource', 'Distance'], axis = 1)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Manual preprocessing, because there is a chance that the maximum value (100 for age and distance, 6000 for capital resource) is not present in a datapoint in the dataset.
    # If that happens, the values can be slightly off, which is fixed by doing it manually
    for i in range(len(x_train_manual.Age)):
        x_train_manual.loc[i, 'Age'] /= 100
        x_test_manual.loc[i, 'Age'] /= 100
        x_train_manual.loc[i, 'Distance'] /= 100
        x_test_manual.loc[i, 'Distance'] /= 100
        x_train_manual.loc[i, 'Resource'] /= 10000
        x_test_manual.loc[i, 'Resource'] /= 10000

    x_train = np.concatenate([x_train_manual, x_train], axis = 1)
    x_test = np.concatenate([x_test_manual, x_test], axis = 1)

    #pd.DataFrame(x_train, columns = x_train_copy.columns).to_excel('DataRes/preprocessedTrain.xlsx')
    #pd.DataFrame(x_test, columns = x_test_copy.columns).to_excel('DataRes/preprocessedTest.xlsx')

    return (x_train, x_test, y_train, y_test)
    
def findHyperParameters(train, test, modelType, testingType):
    x_train, x_test, y_train, y_test = manualScaleSplit(train, test)
    param_grid = {}
    grid = []
    models = []

    if modelType == "MLP":
        models.append(MLPClassifier(hidden_layer_sizes=(24, 10, 3), activation = 'logistic', max_iter = 3000))
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1], 'learning_rate_init': [0.0001, 0.001, 0.01, 0.1]}

    if modelType == "Random_Forest":
        models.append(RandomForestClassifier())
        param_grid = {'max_depth': list(range(1, 30)), 'min_samples_split': list(range(2, 10)), 'max_leaf_nodes': list(range(2, 20))}

    if modelType == "XGBoost": 
        models.append(XGBClassifier(objective ='reg:squarederror', verbosity = 0))       
        param_grid = {'max_depth': list(range(1, 30)), 'learning_rate': np.arange(0.05, 0.55, 0.05), 'gamma': np.arange(0.05, 0.55, 0.05)}

    for j, model in enumerate(models):
        if modelType == "MLP":
            grid = HalvingGridSearchCV(model, param_grid, cv=5, scoring = 'accuracy', factor=2, resource='batch_size', max_resources = 200)
        else:
            grid = HalvingGridSearchCV(model, param_grid, cv=5, scoring = 'accuracy', factor=2, resource='n_estimators', max_resources = 30)
        result = grid.fit(x_train, y_train)
        df = pd.DataFrame(result.cv_results_)
        df.to_excel('DataRes/hyperparameters/' + modelType + testingType + str(j) + '.xlsx')
        for i in range(len(df.iter)):
            if df.loc[i, 'rank_test_score'] == 1:
                print(df.loc[i, 'params'])

# Function that fits the models to the test data
# It prints analysis data, as well as returning that data
def fitModels(models, x_train, x_test, y_train, y_test):
    predicts = []
    accuracies = []

    for model in models:
        model.fit(x_train, y_train)
        predict = model.predict(x_test)
        accuracy = accuracy_score(y_test, predict)
        print("Accuracy score: " + str(accuracy))
        print("Confusion matrix: \n" + str(confusion_matrix(y_test, predict)))
        predicts.append(predict)
        accuracies.append(accuracy)

    return (predicts, accuracies)