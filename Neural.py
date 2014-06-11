import sklearn
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from math import sqrt
import numpy as np
from numpy import dot, transpose, multiply
from layers import *
from functions import *
import ipdb
import copy

epsilon = 10**(-8)

'''
As grid_search.GridSearchCV uses set_params to apply parameter setting to estimators, 
it is essential that calling set_params has the same effect as setting parameters using 
the __init__ method. The easiest and recommended way to accomplish this is to not do any 
parameter validation in ``__init__``. All logic behind estimator parameters, like 
translating string arguments into functions, should be done in fit.

http://scikit-learn.org/stable/developers/index.html#cloning

1. Change NN so that it follows the rule above
2. Change DotLayer so that it follows the rule above
'''


class NN(BaseEstimator):

  def __init__(self,layers = [], type='R', lr=0.01, n_iter=1000, noisy=None, verbose=False):
    
    self.layers = copy.deepcopy(layers)
    self.type = type
    self.lr = lr
    self.n_iter = n_iter
    self.noisy = noisy
    self.verbose = verbose

    self.dataset_index = 0
    self.epoch_index = 0

    # Classification problem
    if type == 'C':
      self.error_func = CrossEntropyError
    if type == 'R':
      self.error_func = SquaredError
    else:
      assert "Unsupported type. 'C' (for classification) or 'R' for regression"

  def add_layer(self, layer):
    self.layers.append(layer)

  def predict(self, X):
    current_results = X
    for layer in self.layers:
      current_results = layer.forward(current_results)

    if self.type == 'C':
      current_results = coalesce(current_results)
    return current_results

  def predict_proba(self, X):
    current_results = X
    for layer in self.layers:
      current_results = layer.forward(current_results)
    return current_results

  def score(self, X, T):
    Y = self.predict(X)
    return 1 / (self.error(X,T) + 1)

  def error(self, X, T):
    Y = self.predict_proba(X)
    return self.error_func.func(Y, T)

  def grab_next(self, X, T):
    X_next = X[self.dataset_index: self.dataset_index+1]
    T_next = T[self.dataset_index: self.dataset_index+1]

    self.dataset_index += 1

    if (len(X_next) == 0) or (len(T_next) == 0):
      X_next = X[0:1]
      T_next = T[0:1]
      self.epoch_index += 1
      self.dataset_index = 0

    return X_next, T_next

  # Maybe switch the order of north partial and 
  # west partial if we ever build the zmq version of this
  # For this version switching the order doesn't add any speedup
  def update(self, X, T):
    #import ipdb; ipdb.set_trace()
    X_next, T_next = self.grab_next(X,T)
    #import ipdb; ipdb.set_trace()
    if self.noisy:
      X_next += self.noisy*np.random.standard_normal(X_next.shape)
      #T_next += self.noisy*np.random.standard_normal(T_next.shape)

    Y_next = self.predict_proba(X_next)
    cur_partial = self.error_func.grad(Y_next,T_next)*self.lr
    rev_layers = reversed(self.layers)
    for (index,layer) in enumerate(rev_layers):

      next_partial = layer.west_partial(cur_partial)
      layer.north_partial(cur_partial)
      cur_partial = next_partial
      '''
      if index != (len(self.layers) - 1):
        cur_partial = layer.west_partial(cur_partial)
      '''
      
  def fit(self, X, T):
    #ipdb.set_trace()
    for i in xrange(self.n_iter):
      
      if (self.verbose and (i % 100 == 0)):
        print "Error: %f" % self.error(X,T)
      self.update(X,T)

  def list_delta_iterators(self):
    return map(lambda x: x.delta_iterator(), self.layers)

  def analytic_gradient(self, X, T):
    Y = self.predict_proba(X)
    cur_partial = self.error_func.grad(Y,T)
    rev_layers = reversed(self.layers)
    gradient = []
    for layer in rev_layers:

      #Compute the partial north, and west
      next_partial = layer.west_partial(cur_partial)
      layer_grad = layer.north_partial(cur_partial)      
      cur_partial = next_partial

      gradient.append(layer_grad)

    return list(reversed(gradient))

  def numerical_gradient(self, X, T):
    J = self.error(X,T)
    layer_iterators = self.list_delta_iterators()
    all_gradients = []
    # Loop over layers
    for layer in layer_iterators:
      layer_deltas = []
      # Loop over W,b in layer
      for weight_structure in layer:
        grad = np.zeros(weight_structure.shape)
        # Loop over elem in W (or equivalent parameter holder)
        for elem in weight_structure:
          elem[...] = elem + epsilon
          J_up = self.error(X,T)
          elem[...] = elem - epsilon
          grad[weight_structure.multi_index] = (J_up - J)/epsilon

        layer_deltas.append(grad)

      all_gradients.append(layer_deltas)

    return all_gradients

  '''
  def fit_converge(self, X_train, T_train, X_test, T_test):
    
    #prev_train_error = self.error(X_train, T_train)
    prev_test_error = self.error(X_test, T_test)
      

    while True:
      for i in range(100):
        self.update(X_train, T_train)

      train_error = self.error(X_train, T_train)
      test_error = self.error(X_test, T_test)
    
      print "Train Error: " + str(train_error)    
      print "Test Error: " + str(test_error)    

      print "Previous Error: " + str(prev_test_error)

      if prev_test_error  < test_error:
        break

      prev_test_error = test_error

  '''
'''

N = NN()
N.add_layer(ConvLayer3(dim=(3,3,3)))
N.add_layer(MeanLayer(dim=(5,5)))

#X = np.random.uniform(size=(7,7))
#T_1 = np.random.uniform(size=(6,6))
#T_2 = np.random.uniform(size=(3,3))

im = Image('lenna').getNumpy()/255.0
im_out = np.random.uniform(size=(102,102))

#X = np.array([im, im])
#T = np.array([im_out, im_out])
#print N.numerical_gradient(X,T_2)
#print N.analytic_gradient(X ,T_2)


print 'Test NN Linear Regression stopping criterion'
## Test Linear Regression
# Get the Boston Regression Data
boston = datasets.load_boston()
X, T = shuffle(boston.data, boston.target)
X = X.astype(np.float32)
T = T.reshape(T.shape[0], 1)

# Preprocess the X data, by using a scaling each feature 
# independently. Scale both the test/train sets
X_scaler = StandardScaler()
X_train, X_test, T_train, T_test = sklearn.cross_validation.train_test_split(X, T, test_size=.2) 

X_train = X_scaler.fit_transform(X_train)
X_test  = X_scaler.transform(X_test)
# Create our linear model
N = NN()
N.add_layer(DotLayer(dim=(13,1)))
N.fit_converge(X_train,T_train, X_test, T_test)

'''