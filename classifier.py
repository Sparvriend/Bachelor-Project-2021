from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import setup

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

    stdScaler = StandardScaler()
    stdScaler.fit(x_train)
    StandardScaler(copy = True, with_mean = True, with_std = True)
    x_train = stdScaler.transform(x_train)
    x_test = stdScaler.transform(x_test)

    return (x_train, x_test, y_train, y_test)

# The parameter search space is yet to be defined for each learning system
def findHyperParameters(train, test, modelType, testingType):
    x_train, x_test, y_train, y_test = manualScaleSplit(train, test)
    param_grid = {}
    grid = []
    models = []

    if modelType == "MLP":
        models = setup.getMLPModels()
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
        print(model)
        print("Accuracy score:\n" + str(accuracy))
        predicts.append(predict)
        accuracies.append(accuracy)

    return (predicts, accuracies)