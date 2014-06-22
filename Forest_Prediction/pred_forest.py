import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import sys
sys.path.insert(0, '..') # Pull in helper file
from helper import *

df_train = pd.read_table('train.csv', sep=",")
df_test  = pd.read_table('test.csv', sep=",")

# Should really take the angles and break them out
# into their sin, cos portions

def map_cos(angle_degrees):
  return np.cos(angle_degrees*np.pi/180)

def map_sin(angle_degrees):
  return np.sin(angle_degrees*np.pi/180)


mapper = Mapper()
mapper.add_index('Id')
mapper.add_X('Elevation')

mapper.add_X('Aspect', map_cos)
mapper.add_X('Aspect', map_sin)
mapper.add_X('Slope', map_cos)
mapper.add_X('Slope', map_sin)

mapper.add_X(['Horizontal_Distance_To_Hydrology', 
              'Vertical_Distance_To_Hydrology',
              'Horizontal_Distance_To_Roadways',
              'Horizontal_Distance_To_Fire_Points'])


mapper.add_X(['Hillshade_9am',
              'Hillshade_Noon',
              'Hillshade_3pm'])

soil_cols = filter(lambda x: 'Soil_Type' in x, df_train.columns)
mapper.add_X(soil_cols)
wilderness_cols = filter(lambda x: 'Wilderness_Area' in x, df_train.columns)
mapper.add_X(wilderness_cols)

mapper.add_Y('Cover_Type')

org = Org()
org.mapper = mapper
R = RandomForestClassifier(n_estimators=300, n_jobs=-1)
E = ExtraTreesClassifier(n_estimators=300, n_jobs=-1)
org.models = [R,E]
#print org.cross_validate(df_train)
org.fit(df_train)

results = org.predict(df_test, as_df=True)
org.write_to_file(results, ['Id', 'Cover_Type'], ['RandomForestClassifier', 'ExtraTreesClassifier'])
#first_result = results[0]
#first_result.columns = ['Id', 'Cover_Type']
#first_result.to_csv('results3_mine.csv', index=False)
