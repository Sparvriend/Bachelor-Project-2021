from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
import pandas as pd

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
def findHyperParameters(train, test, models, modelType):
    x_train, x_test, y_train, y_test = manualScaleSplit(train, test)
    param_grid = {}
    grid = []

    if modelType == "MLP":
        param_grid = {}
        grid = HalvingGridSearchCV(model, param_grid, cv=5, scoring = 'accuracy', factor=2, resource='', max_resources=)
    if modelType == "Random_Forest":
        param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
        grid = HalvingGridSearchCV(model, param_grid, cv=5, scoring = 'accuracy', factor=2, resource='n_estimators', max_resources=)
    if modelType == "XGBoost":
        param_grid = {}
        grid = HalvingGridSearchCV(model, param_grid, cv=5, scoring = 'accuracy', factor=2, resource='', max_resources=)

    for model in models:
        result = grid.fit(x_train, y_train)
        df = pd.DataFrame(result.cv_results_)
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