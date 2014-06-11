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

mapper.add_Y('Cover_Type', LabelBinarizer())

#X,Y = mapper.fit_transform(df_train)

org = Org()
org.mapper = mapper

# Create the model
N = NN(n_iter = 100000, type='C', noisy=.1)
N.add_layer(DotLayer(dim=(54,50)))
N.add_layer(TanhLayer())
N.add_layer(DotLayer(dim=(50,7)))
N.add_layer(SigLayer())

org.models = [N]
print org.cross_validate(df_train, ravel=False)
'''
org.fit(df_train, ravel=False)

results = org.predict(df_test, as_df=False)

first_result = results[0]
l = mapper.dict_mapping[('Cover_Type',)]['pipeline'][0]
guesses = l.inverse_transform(first_result[:, 1:])

index = col(first_result[:,0])
guesses = col(guesses)
output = np.hstack([index, guesses]).astype(int)

result_df = pd.DataFrame(output)
result_df.columns = ['Id', 'Cover_Type']
result_df.to_csv('results_nn_1.csv', index=False)
'''