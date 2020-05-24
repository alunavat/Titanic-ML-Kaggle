import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics


def convert_to_dummy(train, column):
    dummy_variable_1 = pd.get_dummies(train[column])
    train = pd.concat([train, dummy_variable_1], axis=1)
    train.drop(column, axis=1, inplace=True)
    return train

#Import Data
train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')
test1 = pd.read_csv('../test.csv')


#Removed unwanted Columns
train.drop(['PassengerId','Ticket','Cabin','Name','Parch','SibSp','Fare'],axis=1,inplace=True)
test.drop(['PassengerId','Ticket','Cabin','Name','Parch','SibSp','Fare'],axis=1,inplace=True)

#Replace NaN values in Age by Mean
avg_age = train['Age'].astype(float).mean(axis=0)
train['Age'].replace(np.NaN, avg_age, inplace=True)
avg_age = test['Age'].astype(float).mean(axis=0)
test['Age'].replace(np.NaN, avg_age, inplace=True)

#Convert All variables into one hot encoding

train = convert_to_dummy(train,"Embarked")
train = convert_to_dummy(train,"Sex")
train = convert_to_dummy(train,"Pclass")
test = convert_to_dummy(test,"Pclass")
test = convert_to_dummy(test,"Sex")
test = convert_to_dummy(test,"Embarked")

#Binning Data
bins = np.linspace(min(train["Age"]), max(train["Age"]), 3)
group_names = ['Under 16', 'Above 16 and Under 48']
train['Age-binned'] = pd.cut(train['Age'], bins, labels=group_names, include_lowest=True )
dummy_variable_1 = pd.get_dummies(train["Age-binned"])
train = pd.concat([train, dummy_variable_1], axis=1)
train.drop(["Age","Age-binned"], axis = 1, inplace=True)
bins = np.linspace(min(test["Age"]), max(test["Age"]), 3)
group_names = ['Under 16', 'Above 16 and Under 48']
test['Age-binned'] = pd.cut(test['Age'], bins, labels=group_names, include_lowest=True )
dummy_variable_1 = pd.get_dummies(test["Age-binned"])
test = pd.concat([test, dummy_variable_1], axis=1)
test.drop(["Age","Age-binned"], axis = 1, inplace=True)



#Create Model and split data for testing
Z = train.drop(['Survived'],axis=1)
Z_test = test
# X_train, X_test, y_train, y_test = train_test_split(Z, train[['Survived']], test_size=0.3,random_state=0)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
scores = cross_val_score(clf, Z, train[['Survived']].values.ravel(), cv=20)
print ("Cross-validated scores:", np.mean(scores))

#Fir final model with everything and test on actual test data
model = clf.fit(Z, train[['Survived']])
predictions = clf.predict(Z_test)
predictions = pd.DataFrame( {'Survived' :predictions})
test_result = pd.concat([test1['PassengerId'],predictions],axis=1)
test_result.to_csv('../gender_submission.csv')