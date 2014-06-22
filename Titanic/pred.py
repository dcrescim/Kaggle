import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sys
sys.path.insert(0, '..') # Pull in helper file
sys.path.insert(0, '../../DFMapper')
from helper import *
from DFMapper import *

df_train = pd.read_table("train.csv", sep=",")
df_test = pd.read_table("test.csv", sep=",")

mapper = DFMapper()
mapper.add_X('Pclass', LabelBinarizer())
mapper.add_X('Sex', [lambda x: x == "male", LabelBinarizer()])
mapper.add_X('Age', [Imputer()])
mapper.add_X('SibSp')
mapper.add_X('Parch')
mapper.add_X('Fare', [Imputer()])
mapper.add_Y('Survived')
mapper.add_X('Embarked', [LabelEncoder(), LabelBinarizer()])
mapper.add_index('PassengerId')
#mapper.add_option('explode', 2)

org = Org()
org.mapper = mapper
org.models = [RandomForestClassifier(n_estimators=100)]
#print org.cross_validate(df_train)

org.fit(df_train)
results = org.predict(df_test, as_df=True)
org.write_to_file(results, ['PassengerId', 'Survived'], ['RandomForestClassifier'])
