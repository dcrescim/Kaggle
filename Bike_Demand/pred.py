import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

import sys
sys.path.insert(0, '..') # Pull in helper file
sys.path.insert(0, '../../DFMapper')
from helper import *
from DFMapper import *


def time_int(times):
  vals = times.applymap(lambda x: int(x.split()[1].split(':')[0]))
  return col(vals.values)

def time_to_cos(times):
  times = times.applymap(lambda x: int(x.split()[1].split(':')[0]))
  times = times.applymap(lambda x: np.cos(x/24.0))
  return col(times.values)

def time_to_sin(times):
  times = times.applymap(lambda x: int(x.split()[1].split(':')[0]))
  times = times.applymap(lambda x: np.sin(x/24.0))
  return col(times.values)


df_train = pd.read_table('train.csv', sep=",")
df_test  = pd.read_table('test.csv', sep=",")
# Should really take the angles and break them out
# into their sin, cos portions

mapper = DFMapper()
mapper.add_index('datetime')
mapper.add_X('datetime', [time_int, LabelBinarizer()], as_col=False)
mapper.add_X('datetime', time_to_cos, as_col=False)
mapper.add_X('datetime', time_to_sin, as_col=False)

mapper.add_X('season', LabelBinarizer())
mapper.add_X('holiday')
mapper.add_X('workingday')
mapper.add_X('weather', LabelBinarizer())
# Could break temp out as average relative to season
mapper.add_X('temp')
mapper.add_X('atemp')
mapper.add_X('humidity')
mapper.add_X('windspeed')

mapper.add_Y('count')
mapper.add_option('explode', 2)

org = Org()
org.mapper=mapper
org.models = [RandomForestRegressor(n_estimators=10)]
org.fit(df_train)

#print org.cross_validate(df_train)
results = org.predict(df_test, as_df=True)
results['RandomForestRegressor'] = results['RandomForestRegressor'].map(lambda x: '%.3f' % x)
org.write_to_file(results, ['datetime', 'count'], ['RandomForestRegressor'])

