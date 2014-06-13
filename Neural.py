
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from math import sqrt
import numpy as np
from numpy import dot, transpose, multiply
from layers import *
from functions import *
import ipdb
import copy
import pickle
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

class NNBase(BaseEstimator):

  def __init__(self,layers = [], lr=0.01, n_iter=1000, noisy=None, verbose=False, file=None):
    self.layers = copy.deepcopy(layers)
    self.lr = lr
    self.n_iter = n_iter
    self.noisy = noisy
    self.verbose = verbose
    self.file = file
    self.dataset_index = 0
    self.epoch_index = 0
    self.test_index = 0

  def add_layer(self, layer):
    self.layers.append(layer)

  # Returns an output which matches the internal algorithm
  # Returns real numbers for regression, and probabilities for classification
  def _predict(self, X):
    current_results = X
    for layer in self.layers:
      current_results = layer.forward(current_results)

    return current_results

  def _fit(self, X, T):

    for i in xrange(self.n_iter):
      if (self.verbose and (i % 100 == 0)):
        print "Error: %f" % self._error(X,T)
      self._update(X,T)

  def _score(self, X, T):
    Y = self._predict(X)
    return self.score_func.func(Y,T)

  def _error(self, X, T):
    Y = self._predict(X)
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
  def _update(self, X, T):
    X_next, T_next = self.grab_next(X,T)
    if self.noisy:
      X_next += self.noisy*np.random.standard_normal(X_next.shape)
    
    Y_next = self._predict(X_next)
    cur_partial = self.error_func.grad(Y_next,T_next)*self.lr
    rev_layers = reversed(self.layers)
    for (index,layer) in enumerate(rev_layers):
      next_partial = layer.west_partial(cur_partial)
      layer.north_partial(cur_partial)
      cur_partial = next_partial
      

  def list_delta_iterators(self):
    return map(lambda x: x.delta_iterator(), self.layers)

  def _analytic_gradient(self, X, T):
    Y = self._predict(X)
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

  def _numerical_gradient(self, X, T):
    J = self._error(X,T)
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
          J_up = self._error(X,T)
          elem[...] = elem - epsilon
          grad[weight_structure.multi_index] = (J_up - J)/epsilon

        layer_deltas.append(grad)

      all_gradients.append(layer_deltas)

    return all_gradients

  def pickle(self, filename):
    pass
  def unpickle(self):
    pass

class NN_Classifier(NNBase):

  def __init__(self,layers = [], lr=0.01, n_iter=1000, noisy=None, verbose=False):
    
    super(NN_Classifier, self).__init__(layers=layers, lr=lr, n_iter=n_iter, noisy=noisy, verbose=verbose)
    self.type = 'C'
    self.error_func = CrossEntropyError
    self.accuracy_error = AccuracyError
    self.label_binarizer = LabelBinarizer()
    #self.score_func = Figure out what goes here 

  def predict(self, X):
    current_results = NNBase._predict(self, X)
    # Break out the loop here
    current_results = coalesce(current_results)
    return self.label_binarizer.inverse_transform(current_results)

  def predict_proba(self, X):
    return NNBase._predict(self, X)

  def fit(self, X, T):
    T_impl = self.label_binarizer.fit_transform(T)
    NNBase._fit(self, X, T_impl)

  def error_accuracy(self, X, T):
    Y = self.predict(X)
    return self.accuracy_error.func(Y, T)

  def error(self, X, T):
    T_impl = self.label_binarizer.transform(T)
    return NNBase._error(self, X, T_impl)

  def score(self, X, T):
    return 1 - self.error_accuracy(X,T)

  def analytic_gradient(self, X, T):
    T_impl = self.label_binarizer.transform(T)
    return NNBase._analytic_gradient(self, X, T_impl)

  def numerical_gradient(self, X, T):
    T_impl = self.label_binarizer.transform(T)
    return NNBase._numerical_gradient(self, X, T_impl)

class NN_Regressor(NNBase):

  def __init__(self,layers = [], lr=0.01, n_iter=1000, noisy=None, verbose=False):
    
    super(NN_Regressor, self).__init__(layers=layers, lr=lr, n_iter=n_iter, noisy=noisy, verbose=verbose)
    self.type = 'R'
    self.error_func = SquaredError
    #self.score_func = Figure out what goes here 
  
  def predict(self, X):
    return NNBase._predict(self, X)

  def fit(self, X, T):
    return NNBase._fit(self, X, T)

  def score(self, X, T):
    return NNBase._score(self, X, T)

  def error(self, X, T):
    return NNBase._error(self, X, T)

  def analytic_gradient(X,T):
    return NNBase._analytic_gradient(self, X, T)

  def numerical_gradient(X,T):
    return NNBase._numerical_gradient(self, X, T)


# Need to fix this
class NN(object):
  def __init__(self, type='R', layers = [], lr=0.01, n_iter=1000, noisy=None, verbose=False):
    if type == 'R':
      self = NN_Regressor(layers=layers, lr=lr, n_iter=n_iter, noisy=noisy, verbose=verbose)
    elif type == 'C':
      self = NN_Classifier(layers=layers, lr=lr, n_iter=n_iter, noisy=noisy, verbose=verbose)
    else:
      assert "Unsupported type. 'C' (for classification) or 'R' for regression"
