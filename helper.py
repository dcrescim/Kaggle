import numpy as np
import sklearn
import ipdb
import glob
from sklearn.externals import joblib
import pandas as pd

# Takes numpy array, and returns a row array
def row(arr):
  if len(arr.shape) == 1:
    return arr.reshape(1, len(arr))
  return arr

def col(arr):
  if len(arr.shape) == 1:
    return arr.reshape(len(arr), 1)
  return arr

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

      cv = sklearn.cross_validation.ShuffleSplit(len(X), n_iter=6, test_size=.2, random_state=self.random_seed)
      scores = sklearn.cross_validation.cross_val_score(model, X, Y, cv=cv)
      output.append(scores.mean())      
    return output

  def pickle(self):
    for model in self.models:
      joblib.dump(model, type(model).__name__ + ".pkl", compress=9)
   
  def unpickle(self):
    self.models = []
    models = glob.glob("*.pkl")
    for model in models:
      clf = joblib.load(model)
      self.models.append(clf)
    # Get all files ending in .pkl
    # Load em up

  def fit(self, df, ravel=True):
    X,Y = self.mapper.fit_transform(df)
    if ravel:
      Y = np.ravel(Y)

    for model in self.models:
      model.fit(X, Y)
  
  def predict(self,df, as_df=False):
    #import ipdb; ipdb.set_trace()
    X, _ = self.mapper.transform(df)
    output = []
    for model in self.models:
      results = model.predict(X)
      output.append(col(results))

    output.insert(0, self.mapper.index)
    final = np.hstack(output)
    if as_df:
      final = pd.DataFrame(final)
      column_names = ['index']
      for model in self.models:
        name = type(model).__name__
        column_names.append(name)
      final.columns = column_names
    return final



  def write_to_file(self,df, column_names, model_names):
    #import ipdb; ipdb.set_trace()
    for model in model_names:
      small_df = df[['index', model]]
      small_df.columns = column_names
      small_df.to_csv(model +'.csv', index=False)






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