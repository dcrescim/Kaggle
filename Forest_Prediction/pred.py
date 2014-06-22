import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import Imputer, LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sys
sys.path.insert(0, '..') # Pull in helper file
sys.path.insert(0, '../../DFMapper') # Pull in DFMapper

from helper import *
from DFMapper import *
import pickle

df_train = pd.read_table('train.csv', sep=",")
df_test  = pd.read_table('test.csv', sep=",")

# Should really take the angles and break them out
# into their sin, cos portions
mapper = DFMapper()
mapper.add_index('Id')
mapper.add_X('Elevation')
mapper.add_X('Aspect')
mapper.add_X('Slope')
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

X, Y = mapper.fit_transform(df_train)
bigX = np.hstack([X,explode(X,2)])

E = ExtraTreesClassifier(n_estimators=400, n_jobs=-1)
E.fit(bigX, Y)
print "finished training"

testX, _ = mapper.transform(df_test)
length = len(testX)
div = length/30
big_results = []
for i in range(31):
  print i
  small_x = testX[i*div: (i+1)*div]
  additional = explode(small_x,2)
  small_x = np.hstack([small_x, additional])
  #results = l.predict(small_x)
  results = E.predict(small_x)
  big_results.append(results)

big_results = map(col, big_results)
here = np.vstack(big_results)

result_df = df_test[['Id']]
result_df['Cover_Type'] = here
result_df.to_csv('extra_forest_explode.csv', index=False)





























































"""

myorg = Org()
myorg.mapper = mymapper
# Create the model
N = NN_Classifier(epochs=1, lr=0.001)
N.add_layer(DotLayer(dim=(54,1000)))
N.add_layer(TanhLayer())
N.add_layer(DotLayer(dim=(1000,7)))
N.add_layer(SigLayer())
myorg.models = [N]
#print org.cross_validate(df_train, ravel=False)
myorg.unpickle()
#myorg.models[0].epochs= 10
myorg.fit(df_train)
print "finished training"

print myorg.models[0].score(myX, myY)
print myorg.models[0].error(myX, myY)
#myresults = myorg.predict(df_test, as_df=True)
myorg.pickle()
"""
"""
'''
========================================
========================================
========================================
'''

mapper = Mapper()
mapper.add_index('Id')
mapper.add_X('Elevation')

mapper.add_X('Aspect')
mapper.add_X('Slope')

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


X,Y = mapper.fit_transform(df_train)
#X_test, _ = mapper.fit_transform(df_test)

org = Org()
org.mapper = mapper
R = RandomForestClassifier(n_estimators=150, n_jobs=-1)
org.models = [R]
#print org.cross_validate(df_train)
org.fit(df_train)
R.score(X,Y)
results = org.predict(df_test, as_df=True)
#org.write_to_file(results, ['Id', 'Cover_Type'], ['RandomForestClassifier'])






final = pd.merge(myresults, results, on = 'index')






"""