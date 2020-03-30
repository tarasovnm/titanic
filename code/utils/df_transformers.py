import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.preprocessing import LabelEncoder

class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column selection
    
    Allows to select columns by name from pandas dataframes in scikit-learn
    pipelines.
    
    Parameters
    ----------
    columns : list of str, names of the dataframe columns to select
        Default: [] 
    
    """
    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):
        """ Selects columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
            contains selected columns of X      
        """
        trans = X[self.columns].copy() 
        return trans
    
    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self


class DataFrameFunctionTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application
    
    Parameters
    ----------
    impute : Boolean, default False
        
    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise 
        an array of the form [n_elements, 1]
    
    """
    
    def __init__(self, func, impute = False):
        self.func = func
        self.impute = impute
        self.series = pd.Series() 

    def transform(self, X, **transformparams):
        """ Transforms a DataFrame
        
        Parameters
        ----------
        X : DataFrame
            
        Returns
        ----------
        trans : pandas DataFrame
            Transformation of X 
        """
        
        if self.impute:
            trans = pd.DataFrame(X).fillna(self.series).copy()
        else:
            trans = pd.DataFrame(X).apply(self.func).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Fixes the values to impute or does nothing
        
        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement
                
        Returns
        ----------
        self  
        """
        
        if self.impute:
            self.series = pd.DataFrame(X).apply(self.func).copy()
        return self


class DataFrameFeatureUnion(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that unites several DataFrame transformers
    
    Fit several DataFrame transformers and provides a concatenated
    Data Frame
    
    Parameters
    ----------
    list_of_transformers : list of DataFrameTransformers
        
    """ 
    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers
        
    def transform(self, X, **transformparamn):
        """ Applies the fitted transformers on a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
        
        Returns
        ----------
        concatted :  pandas DataFrame
        
        """
        
        concatted = pd.concat([transformer.transform(X)
                            for transformer in
                            self.fitted_transformers_], axis=1).copy()
        return concatted
    
    def fit(self, X, y=None, **fitparams):
        """ Fits several DataFrame Transformers
        
        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement
        
        Returns
        ----------
        self : object
        """
        
        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self


class ToDummiesTransformer(BaseEstimator, TransformerMixin):
    """ A Dataframe transformer that provide dummy variable encoding
    """
    
    def transform(self, X, **transformparams):
        """ Returns a dummy variable encoded version of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
        
        Returns
        ----------
        trans : pandas DataFrame
        
        """
    
        trans = pd.get_dummies(X, drop_first=True, dummy_na=True, sparse=True,).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Do nothing operation
        
        Returns
        ----------
        self : object
        """
        return self


class DropAllZeroTrainColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides dropping all-zero columns
    """

    def transform(self, X, **transformparams):
        """ Drops certain all-zero columns of X
        
        Parameters
        ----------
        X : DataFrame
        
        Returns
        ----------
        trans : DataFrame
        """
        
        trans = X.drop(self.cols_, axis=1).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Determines the all-zero columns of X
        
        Parameters
        ----------
        X : DataFrame
        y : not used
        
        Returns
        ----------
        self : object
        """
        
        self.cols_ = X.columns[(X==0).all()]
        return self


class BinaryColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides binary columns encoding
    
    Encodes binary columns by 0 and 1 from pandas dataframes in scikit-learn
    pipelines.

    Parameters
    ----------
    bin_dict : dictionary for mapping values in columns
    fill_na : boolean, true if need to fill missing values
        Default: True
    na_value : value to fill missing values
        Default: 0 
    
    """

    def __init__(self, bin_dict, fill_na = True, na_value = 0):
        self.bin_dict = bin_dict
        self.fill_na = fill_na
        self.na_value = na_value

    def transform(self, X, **transform_params):
        """ Encodes binary columns by 0 and 1
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
     
        """
        trans = X.copy()

        for col in trans.columns:
            if trans.dtypes[col] == np.object:
                trans[col] = trans[col].map(self.bin_dict)

        if self.fill_na:
            trans = trans.fillna(self.na_value)

        return trans
    
    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self


class LabelTransformer(BaseEstimator, TransformerMixin):
    """ A Dataframe transformer that provide label encoding
    """
    
    def transform(self, X, **transformparams):
        """ Returns a label encoded version of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
        
        Returns
        ----------
        trans : pandas DataFrame
        
        """
    
        trans = X.copy()

        for col in trans.columns:
            le = LabelEncoder()
            trans[col] = le.fit_transform(trans[col])

        return trans

    def fit(self, X, y=None, **fitparams):
        """ Do nothing operation
        
        Returns
        ----------
        self : object
        """
        return self


class OrderedColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides ordered columns encoding
    
    Encodes ordered columns by indeces from list

    Parameters
    ----------
    order_list : list for mapping values in columns to indeces
        Default: []
    
    """

    def __init__(self, order_list = []):
        self.order_list = order_list

    def transform(self, X, **transform_params):
        """ Encodes ordered columns by indeces
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
     
        """
        trans = X.copy()

        if len(self.order_list) > 0:
            for i in range(1, len(self.order_list) + 1):
                col = trans.columns[i-1]
                ord_order_dict = {i : j + 1 for j, i in enumerate(self.order_list[i-1])}
                trans[col] = trans[col].map(ord_order_dict)
        else:
            for i in range(1, trans.shape[1] + 1):
                col = trans.columns[i-1]
                values = list(set(list(trans[col].dropna().unique())))
                ord_order_dict = {i : j + 1 for j, i in enumerate(sorted(values))}
                trans[col] = trans[col].map(ord_order_dict)

        return trans
    
    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self

class FillerFunctionTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application 
        for filling missing values
    
    Parameters
    ----------
    impute : Boolean, default False
        
    func : name of the function as stinrg
           variants: 'mean', 'median', 'value'
    
    """
    
    def __init__(self, column, func, value_to_fill=0):
        self.column = column
        self.func = func
        self.value_to_fill = value_to_fill

    def transform(self, X, **transformparams):
        """ Transforms a DataFrame
        
        Parameters
        ----------
        X : DataFrame
            
        Returns
        ----------
        trans : pandas DataFrame
            Transformation of X 
        """

        trans = X.copy()
        trans[self.column].fillna(self.value_to_fill, inplace=True)
        return trans

    def fit(self, X, y=None, **fitparams):
        """ Fixes the values to impute or does nothing
        
        Parameters
        ----------
        X : pandas DataFrame
        y : not used, API requirement
                
        Returns
        ----------
        self  
        """
        
        if self.func == 'mean':
          self.value_to_fill = X[self.column].mean()
        elif self.func == 'median':
          self.value_to_fill = X[self.column].median()

        return self


class RemoveColumnsTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer that provides column removing
    
    Allows to remove columns by name from pandas dataframes in scikit-learn
    pipelines.
    
    Parameters
    ----------
    columns : list of str, names of the dataframe columns to remove
        Default: [] 
    
    """
    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):
        """ Removers columns of a DataFrame
        
        Parameters
        ----------
        X : pandas DataFrame
            
        Returns
        ----------
        
        trans : pandas DataFrame
            contains selected columns of X      
        """
        trans = X.copy() 
        trans.drop(self.columns, axis=1, inplace=True)
        return trans
    
    def fit(self, X, y=None, **fit_params):
        """ Do nothing function
        
        Parameters
        ----------
        X : pandas DataFrame
        y : default None
                
        
        Returns
        ----------
        self  
        """
        return self