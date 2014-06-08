
from Neural import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print 'Simple Integration Test 1'
# Run the normal Test
N = NN()
N.add_layer(DotLayer(dim=(2,2)))
N.add_layer(PureLayer(mapping=Tanh))
X = np.array([.8,.1,3,4]).reshape(2,2)
T = np.array([0,1,3,4]).reshape(2,2)

N.fit(X,T)

print '---------------------------------------'


print 'Test NN Linear Regression on Boston Housing Dataset'
## Test Linear Regression
# Get the Boston Regression Data
boston = datasets.load_boston()
X, T = shuffle(boston.data, boston.target)
X = X.astype(np.float32)
T = T.reshape(T.shape[0], 1)

# Preprocess the X data, by using a scaling each feature 
# independently. Scale both the test/train sets
X_scaler = StandardScaler()
X = X_scaler.fit_transform(X) 

# Create our linear model
N = NN()
N.add_layer(DotLayer(dim=(13,1)))
N.fit(X,T)

print '---------------------------------------'


print 'Test NN Logistic Regression on Iris Dataset'

# Get the Iris Classification Data
iris = datasets.load_iris()
X, T = shuffle(iris.data, iris.target)
X = X.astype(np.float32)
T = T.reshape(T.shape[0], 1)

# Preprocess the X data, by using a scaling each feature 
# independently. Scale both the test/train sets
X_scaler = StandardScaler()
X = X_scaler.fit_transform(X)

# Turn our T into a binary matrix
Encoder = OneHotEncoder()
T = Encoder.fit_transform(T).toarray()

N = NN(type='C')
N.add_layer(DotLayer(dim=(4,3)))
N.add_layer(PureLayer(mapping=Sig))
N.fit(X,T)

print '---------------------------------------'

print 'Test NN on Boston Housing Dataset 2 layers'
## Test Linear Regression
# Get the Boston Regression Data
boston = datasets.load_boston()
X, T = shuffle(boston.data, boston.target)
X = X.astype(np.float32)
T = T.reshape(T.shape[0], 1)

# Preprocess the X data, by using a scaling each feature 
# independently. Scale both the test/train sets
X_scaler = StandardScaler()
X = X_scaler.fit_transform(X) 

# Create our linear model
N = NN()
N.add_layer(DotLayer(dim=(13,5)))
N.add_layer(PureLayer(mapping=Tanh))
N.add_layer(DotLayer(dim=(5,1)))
N.fit(X,T)


print '---------------------------------------'


