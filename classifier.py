from sklearn.metrics import accuracy_score
import numpy as np

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

def onlyTest(trainedModel, testSet, scaler):
    x_test, y_test = scaleData(testSet, scaler)
    predict = trainedModel.predict(x_test)
    accuracy = accuracy_score(y_test, predict)
    print("Accuracy score: " + str(accuracy))

    return (predict, accuracy)

def trainModel(trainSet, model, scaler):
    x_train, y_train = scaleData(trainSet, scaler)
    return model.fit(x_train, y_train)