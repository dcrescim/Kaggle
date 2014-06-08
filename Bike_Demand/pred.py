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

df_train = pd.read_table('train.csv', sep=",")
df_test  = pd.read_table('test.csv', sep=",")
# Should really take the angles and break them out
# into their sin, cos portions

mapper = Mapper()
#mapper.add_X('datetime', time_to_circle)

mapper.add_X('season', LabelBinarizer())
mapper.add_X('holiday')
mapper.add_X('workingday')
mapper.add_X('weather', LabelBinarizer())
# Could break temp out as average relative to season
mapper.add_X('temp', StandardScaler())
mapper.add_X('atemp', StandardScaler())
mapper.add_X('humidity')
mapper.add_X('windspeed', StandardScaler())

mapper.add_Y('count')

X,Y = mapper.fit_transform(df_train)
X_test, _ = mapper.fit_transform(df_test)
