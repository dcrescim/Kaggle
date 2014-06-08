import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0, '..') # Pull in helper file
from helper import *

df_train = pd.read_table("train.csv", sep=",")
df_test = pd.read_table("test.csv", sep=",")

mapper = Mapper()
mapper.add_X('Pclass', LabelBinarizer())
mapper.add_X('Sex', [lambda x: x == "male", LabelBinarizer()])
mapper.add_X('Age', [Imputer(), StandardScaler()])
mapper.add_X('SibSp', LabelBinarizer())
mapper.add_X('Parch', LabelBinarizer())
mapper.add_X('Fare', [Imputer(), StandardScaler()])
mapper.add_Y('Survived')
#mapper.add_X('Embarked', [Imputer(strategy='most_frequent'), LabelEncoder(), LabelBinarizer()])
mapper.add_index('PassengerId')

X_train, Y_train = mapper.fit_transform(df_train)
X_test, _ = mapper.fit_transform(df_test)

'''
org = Org()
org.mapper = mapper
org.models = [LogisticRegression(), LinearSVC(), RandomForestClassifier()]
#org.error_func = 
org.fit(df_train)
print org.output
#org.param_search = []


'''