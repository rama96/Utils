""" Module for Pre-Processing of data before inputing the same into the model """

from typing import List
import pandas as pd
from scipy.stats import norm, skew 

from sklearn.preprocessing import MinMaxScaler , StandardScaler, PowerTransformer

from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import PowerTransformer


class FeatureScaler:
    """ A wrapper over the sklearn preprocessing modules for numerical data
    -----------------------------------------------------------------------------------------------------
    Args : 
        min_max_cols - Coolumns which are to be scaled using min_max_scaler
        standard_cols - Coolumns which are to be scaled using standard_scaler
        power_transform_cols - Coolumns which are to be scaled using power_transform
    -----------------------------------------------------------------------------------------------------
    
    -----------------------------------------------------------------------------------------------------
    Usage : 
    >> scaler = FeatureScaler(min_max_cols = ['col1','col2'] , standard_cols = ['col3','col4'])
    >> scaler.fit(df)
    >> df_transformed = scaler.transform(df)
    
    # alternatively you can also use fit transform 
    >> df_transformed = scaler.fit_transform(df)

    
    -----------------------------------------------------------------------------------------------------
    
    """
    def __init__(self , min_max_cols : List = [] , standard_cols : List = [] , power_transform_cols : List = []) -> None:
        
        self.columns = {
            'min_max':min_max_cols,
            'standard_scaler':standard_cols,
            'power_transformer':power_transform_cols
        }
        
        self.scalers = {
            'min_max':MinMaxScaler(),
            'standard_scaler':StandardScaler(),
            'power_transformer':PowerTransformer(),
        }
                    
    def fit(self,df):
        for key in self.scalers.keys():
            cols = self.columns[key]
            if cols :
                self.scalers[key].fit(df[cols])
    

    def transform(self,df):
        
        for key in self.scalers.keys():
            cols = self.columns[key]
            if cols :
                df[cols] = self.scalers[key].transform(df[cols])
        return df
    
    def fit_transform(self,df):
        self.fit(df)
        return self.transform(df)


class CategoricalEncoder:
    """ A wrapper over the sklearn preprocessing modules for categorical data
    -----------------------------------------------------------------------------------------------------
    Args : 
        nominal_cols : columns to be onehotencoded 
    -----------------------------------------------------------------------------------------------------
    
    -----------------------------------------------------------------------------------------------------
    Usage : 
    
    >> enc = CategoricalEncoder(nominal_cols = ['col1','col2'])
    >> enc.fit(df)
    >> df_transformed = enc.transform(df)
    
    # alternatively you can also use fit transform 
    >> df_transformed = enc.fit_transform(df)

    -----------------------------------------------------------------------------------------------------
    

    """
    def __init__(self , ordinal_cols : List = [] , nominal_cols : List = [] , max_categories = 10) -> None:
        
        # Only implemented for OneHotEncoder()
        self.columns = {
        #    'oridinal':ordinal_cols,
            'nominal':nominal_cols,
        }
        self.max_categories = max_categories
        self.encoders = {
        #    'ordinal':LabelEncoder(),
            'nominal':OneHotEncoder(drop='first',max_categories=max_categories),
        }
                    
    def fit(self,df):
        for key in self.encoders.keys():
            cols = self.columns[key]
            if cols :
                self.encoders[key].fit(df[cols])

    def transform(self,df):
        
        for key in self.encoders.keys():
            cols = self.columns[key]
            if cols :
                
                data = self.encoders[key].transform(df[cols]).toarray()
                colnames = self.encoders[key].get_feature_names_out()
                
                _df = pd.DataFrame(data , columns = colnames)
                df = df.drop(cols , axis = 1)
                df = pd.concat([df,_df],axis = 1)

        return df
    
    def fit_transform(self,df):
        self.fit(df)
        return self.transform(df)



