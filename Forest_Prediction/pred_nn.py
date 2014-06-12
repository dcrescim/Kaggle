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
from Neural import *

# Read in the data

df_train = pd.read_table('train.csv', sep=",")
df_test  = pd.read_table('test.csv', sep=",")

# Should really take the angles and break them out
# into their sin, cos portions

mymapper = Mapper()
mymapper.add_index('Id')

mymapper.add_X('Elevation', StandardScaler())
mymapper.add_X('Aspect', StandardScaler())
mymapper.add_X('Slope', StandardScaler())

mymapper.add_X(['Horizontal_Distance_To_Hydrology', 
              'Vertical_Distance_To_Hydrology',
              'Horizontal_Distance_To_Roadways',
              'Horizontal_Distance_To_Fire_Points'], StandardScaler())


mymapper.add_X(['Hillshade_9am',
              'Hillshade_Noon',
              'Hillshade_3pm'], StandardScaler())

soil_cols = filter(lambda x: 'Soil_Type' in x, df_train.columns)
mymapper.add_X(soil_cols)
wilderness_cols = filter(lambda x: 'Wilderness_Area' in x, df_train.columns)
mymapper.add_X(wilderness_cols)

mymapper.add_Y('Cover_Type')

#X,Y = mymapper.fit_transform(df_train)
org = Org()
org.mapper = mymapper

# Create the model
N = NN_Classifier(n_iter = 100)
N.add_layer(DotLayer(dim=(54,50)))
N.add_layer(TanhLayer())
N.add_layer(DotLayer(dim=(50,7)))
N.add_layer(SigLayer())

org.models = [N, LinearSVC()]
#print org.cross_validate(df_train, ravel=False)
org.fit(df_train)
results = org.predict(df_test, as_df=True)

'''
result_df = pd.DataFrame(output)
result_df.columns = ['Id', 'Cover_Type']
result_df.to_csv('results_nn_1.csv', index=False)
'''