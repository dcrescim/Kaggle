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

# Read in the data

df_train = pd.read_table('train.csv', sep=",")
df_test  = pd.read_table('test.csv', sep=",")

# Create the model

N = NN()
N.add_layer(DotLayer(dim=(54,50)))
N.add_layer(PureLayer(mapping=Tanh))
N.add_layer(DotLayer(dim=(50,7)))



# Should really take the angles and break them out
# into their sin, cos portions

mapper = Mapper()
mapper.add_index('Id')

mapper.add_X('Elevation', StandardScaler())
mapper.add_X('Aspect', StandardScaler())
mapper.add_X('Slope', StandardScaler())

mapper.add_X(['Horizontal_Distance_To_Hydrology', 
              'Vertical_Distance_To_Hydrology',
              'Horizontal_Distance_To_Roadways',
              'Horizontal_Distance_To_Fire_Points'], StandardScaler())


mapper.add_X(['Hillshade_9am',
              'Hillshade_Noon',
              'Hillshade_3pm'], StandardScaler())

soil_cols = filter(lambda x: 'Soil_Type' in x, df_train.columns)
mapper.add_X(soil_cols)
wilderness_cols = filter(lambda x: 'Wilderness_Area' in x, df_train.columns)
mapper.add_X(wilderness_cols)

mapper.add_Y('Cover_Type')



org = Org()
org.mapper = mapper




org.models = [RandomForestClassifier(n_estimators=500, n_jobs=-1)]
#print org.cross_validate(df_train)
org.fit(df_train)

results = org.predict(df_test, as_df=True)

first_result = results[0]
first_result.columns = ['Id', 'Cover_Type']
first_result.to_csv('results3_mine.csv', index=False)
