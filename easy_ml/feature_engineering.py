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
    def __init__(self , ordinal_cols : List = [] , nominal_cols : List = [] ) -> None:
        
        # Only implemented for OneHotEncoder()
        self.columns = {
        #    'oridinal':ordinal_cols,
            'nominal':nominal_cols,
        }
        
        self.encoders = {
        #    'ordinal':LabelEncoder(),
            'nominal':OneHotEncoder(drop='first'),
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
                colnames = [i+"_"+ str(k) for i,j in zip(self.encoders[key].feature_names_in_,self.encoders[key].categories_) for k in j[1:] ]
                
                _df = pd.DataFrame(data , columns = colnames)
                df = df.drop(cols , axis = 1)
                df = pd.concat([df,_df],axis = 1)

        return df
    
    def fit_transform(self,df):
        self.fit(df)
        return self.transform(df)



class PreProcess :
    """ A class used to prepare Categorical and Numerical data that is later fed into the model  """
    # %%
    def __init__(self) -> None:
        pass
        
    # %%
    @staticmethod
    def prepare_data(dt:pd.DataFrame) -> pd.DataFrame: 
        """ Function to Normalize numeric Variables and OneHot encode categorical variables """
        
        dt_columns=dt.columns
        for c in dt_columns:
            print(c)
            if dt[c].dtype != "object":
                dt[c] = PreProcess._normalize(dt[c],2)
            else:
                onehot_encoded = PreProcess._onehot(dt[c], c)
                dt = dt.join(onehot_encoded)
                dt = dt.drop([c], axis=1)
        return dt

    @staticmethod
    def convert_to_onehot(df:pd.DataFrame , cols:list) -> pd.DataFrame: 
        """ Function to one hot columns given the categorical columns and data as input """
        
        dt_columns= cols
        for c in dt_columns:
                onehot_encoded = PreProcess._onehot(df[c], c)
                df = df.join(onehot_encoded)
                df = df.drop([c], axis=1)
        return df


    @staticmethod
    def normalize_numerical_cols(df:pd.DataFrame , cols:list) -> pd.DataFrame: 
        """ Function to one hot columns given the categorical columns and data as input """
        
        dt_columns= cols
        for c in dt_columns:
                df[c] = PreProcess._normalize(df[c],2)
        return df

    
    # %%
    @staticmethod
    def _normalize(X_skewed:pd.Series , skew_threshold:int = 2) -> pd.Series:
        """ Helper function which normalizes numeric variables using power transformation """
        
        if skew(X_skewed)>abs(skew_threshold):
            #X_Normalized, m = stats.boxcox(X_skewed)
            pt = PowerTransformer()
            X_Normalized=pt.fit_transform(X_skewed.values.reshape(-1,1))
            return X_Normalized
        else:
            return X_skewed

    # %%
    @staticmethod
    def _onehot(X_column:pd.Series , column_name:str) -> pd.DataFrame:
        """ Helper function that Uses a label encoder which then Assign labels and OneHotEncodes the whole variable"""
        
        #Labe Encoding
        lbl = LabelEncoder() 
        lbl.fit(list(X_column.values)) 
        X_labelencoded = lbl.transform(list(X_column.values))
        #One Hot encoding
        onehot_encoded=pd.get_dummies(X_labelencoded, prefix=column_name)
        #onehot_encoder = OneHotEncoder(sparse=False)
        #onehot_encoded = onehot_encoder.fit_transform(X_labelencoded.reshape(-1,1))
        return onehot_encoded

