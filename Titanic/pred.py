import pandas as pd
import numpy as np
import sklearn

import sys
sys.path.insert(0, '..') # Pull in helper file
from helper import *

from sklearn_pandas import DataFrameMapper
df = pd.read_table("train.csv", sep=",")

'''
del df['PassengerId']
del df['Name']
del df['Ticket']
del df['Cabin']
Y = df['Survived'].values
del df['Survived']

im_age = sklearn.preprocessing.Imputer()
df['Age'] = im_age.fit_transform(col(df['Age'].values))

im_embarked = sklearn.preprocessing.Imputer()
df['embarked'] = im_embarked.fit_transform(col(df['Age'].values))

mapper = DataFrameMapper([
  (['Pclass'], sklearn.preprocessing.LabelBinarizer()),
  (['Sex'], sklearn.preprocessing.LabelBinarizer()),
  (['SibSp'], sklearn.preprocessing.LabelBinarizer()),
  (['Parch'], sklearn.preprocessing.LabelBinarizer()),
  (['Embarked'], sklearn.preprocessing.LabelBinarizer()),
  (['Age'], sklearn.preprocessing.StandardScaler()),
  (['Fare'], sklearn.preprocessing.StandardScaler())
])

X = mapper.fit_transform(df)

clf = sklearn.linear_model.LogisticRegression(C=.1)
clf.fit(X, Y)

# Make testing set
df_test = pd.read_table("test.csv", sep=",")
'''