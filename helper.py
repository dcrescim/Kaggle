from collections import OrderedDict
import numpy as np
import sklearn
import ipdb
import pandas as pd
def isinstance_func(x):
  return hasattr(x, '__call__')

# Takes numpy array, and returns a row array
def row(arr):
  if len(arr.shape) == 1:
    return arr.reshape(1, len(arr))
  return arr

def col(arr):
  if len(arr.shape) == 1:
    return arr.reshape(len(arr), 1)
  return arr

class Mapper:
  def __init__(self):
    self.dict_mapping = OrderedDict()
    self.index = None
  # Key is a column of the original data
  # function list is a list of one of the following
  #   - A class that implements the Transformer API
  #   - A function
  def _add(self, key, function_list, is_X, is_Y, is_index, as_col=True):
    
    if not isinstance(function_list, list):
      function_list = [function_list]

    if isinstance(key, str):
      key = tuple([key])

    if not isinstance(key, tuple):
      key = tuple(key)


    dict_values = {}
    dict_values['pipeline'] = function_list
    dict_values['is_X'] = is_X
    dict_values['is_Y'] = is_Y
    dict_values['is_index'] = is_index
    dict_values['as_col'] = as_col

    self.dict_mapping[key] = dict_values
  
  def add_X(self, key, function_list=[], as_col = True):
    self._add(key, function_list, is_X=True, is_Y=False, is_index=False, as_col=as_col)

  def add_Y(self, key, function_list=[], as_col = True):
    self._add(key, function_list, is_X=False, is_Y=True, is_index=False, as_col=as_col)

  def add_index(self, key, function_list=[], as_col=True):
    self._add(key, function_list, is_X=False, is_Y=False, is_index=True, as_col=as_col)

  def evaluate(self, key, dict_options, df):
    for el in key:
      if (el not in df):
        # If you are missing an X column, this is bad. 
        #   You should find it.
        if dict_options['is_X']:
          ValueError("The column %s is not in your dataframe" % key)
        
        # If you are missing Y columns, that is not a big deal
        #   You could just be transforming the test set.
        if dict_options['is_Y']:
          return None

    if dict_options['as_col']:
      cur_val = col(df[list(key)].values)
    else:
      cur_val = df[key]

    for f in dict_options['pipeline']:
      if isinstance_func(f):
        cur_val = f(cur_val)
      else:
        cur_val = f.fit_transform(cur_val)

    return cur_val


  def fit_transform(self, df):
    results_X = []
    results_Y = []
    for (key, dict_options) in self.dict_mapping.iteritems():
      cur_val = self.evaluate(key,dict_options, df)

      # This occurs when you are trying to evaluate
      # a key that is not in the dataframe
      if cur_val == None:
        continue

      if dict_options['is_X']:
        results_X.append(cur_val)
      if dict_options['is_Y']:
        results_Y.append(cur_val)
      if dict_options['is_index']:
        self.index = cur_val

    # Can't np.hstack an empty list
    if not results_X:
      X_results = np.array([])
    else:
      X_results = np.hstack(results_X)

    if not results_Y:
      Y_results = np.array([])
    else:
      Y_results = np.hstack(results_Y)

    return X_results, Y_results


class Org:
  def __init__(self):
    self.random_seed = 42
    self.mapper = None
    self.models = []
    self.output = []
    
  def cross_validate(self, df, ravel=True):
    X,Y = self.mapper.fit_transform(df)
    
    if ravel:
      Y = np.ravel(Y)

    output = []
    for model in self.models:
      cv = sklearn.cross_validation.ShuffleSplit(len(X), n_iter=2, test_size=.2, random_state=self.random_seed)
      #import ipdb; ipdb.set_trace() 
      scores = sklearn.cross_validation.cross_val_score(model, X, Y, cv=cv)
      #output.append('%0.3f' % scores.mean())
      output.append(scores)
    return output

  def fit(self, df, ravel=True):
    X,Y = self.mapper.fit_transform(df)
    for model in self.models:
      if ravel:
        model.fit(X,np.ravel(Y))
      else:
        model.fit(X, Y)
  def predict(self,df, as_df=False):
    X, _ = self.mapper.fit_transform(df)
    output = []
    for model in self.models:
      results = model.predict(X)
      final = np.hstack([self.mapper.index, col(results)])
      if as_df:
        final = pd.DataFrame(final)
      output.append(final)
    return output





'''
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.fixes import unique
from sklearn.utils import deprecated, column_or_1d



class MyLabelEncoder(BaseEstimator, TransformerMixin):
    """Encode labels with value between 0 and n_classes-1.

    Attributes
    ----------
    `classes_` : array of shape (n_class,)
        Holds the label for each class.

    Examples
    --------
    `LabelEncoder` can be used to normalize labels.

    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6]) #doctest: +ELLIPSIS
    array([0, 0, 1, 2]...)
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"]) #doctest: +ELLIPSIS
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']

    """

    def _check_fitted(self):
        if not hasattr(self, "classes_"):
            raise ValueError("LabelEncoder was not fitted yet.")

    def peel_off_nan(self):
      has_nan_boolean = np.any(np.isnan(self.classes_))
      classes_without_nan = self.classes_[np.logical_not(np.isnan(self.classes_))]
      return classes_without_nan, has_nan_boolean
    def fit(self, y):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = np.unique(y)

        # Pulls out the nans from self.classes, and sets a boolean
        # if it finds any
        self.classes_, self.has_nan = peel_off_nan(self.classes_)
        return self

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        self._check_fitted()

        classes = np.unique(y)
        classes_without_nan, has_nan = peel_off_nan(classes)

        if len(np.intersect1d(classes, self.classes_)) < len(classes) or
          (has_nan and not self.has_nan):
            diff = np.setdiff1d(classes, np.append(self.classes_, self.nan_array_))
            raise ValueError("y contains new labels: %s" % str(diff))
        
        return np.searchsorted(self.classes_, y) 

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        self._check_fitted()

        y = np.asarray(y)
        return self.classes_[y]

'''