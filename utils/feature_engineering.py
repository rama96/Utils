""" Module for Pre-Processing of data before inputing the same into the model """

from typing import List
import pandas as pd
from scipy.stats import norm, skew 

from sklearn.preprocessing import MinMaxScaler , StandardScaler, PowerTransformer

from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import PowerTransformer


class FeatureScaling:
    
    def __init__(self , df : pd.DataFrame = None , min_max_cols : List = [] , standard_cols : List = [] , power_transform_cols : List = []) -> None:
        self.min_max_cols = min_max_cols
        self.standard_cols = standard_cols
        self.power_transform_cols = power_transform_cols
        
        
    def min_max_scaler(self,df) -> pd.DataFrame:
        cols = self.min_max_cols
        if not cols:
            return df
        else :
            min_max = MinMaxScaler()
            df[cols] = pd.DataFrame(min_max.fit_transform(df[cols]) , columns = cols)
            return df
    
    def standard_scaler(self,df) -> pd.DataFrame:
        cols = self.standard_cols
        if not cols:
            return df
        else :
            min_max_scaler = StandardScaler()
            df[cols] = pd.DataFrame(min_max_scaler.fit_transform(df[cols]) , columns = cols)
            return df
        
    def power_transformer(self,df) -> pd.DataFrame:
        cols = self.power_transform_cols
        if not cols:
            return df
        else :
            for col in cols:
                df[col] = self._power_transform(df[col])
            return df
                
    def _power_transform(self , X_skewed:pd.Series , skew_threshold:int = 2) -> pd.Series:
        """ Helper function which normalizes numeric variables using power transformation """
        if skew(X_skewed)>abs(skew_threshold):
            #X_Normalized, m = stats.boxcox(X_skewed)
            pt = PowerTransformer()
            X_Normalized=pt.fit_transform(X_skewed.values.reshape(-1,1))
            return X_Normalized
        else:
            return X_skewed
    
    @property
    def tranform(self):
        df = self.min_max_scaler(self,self.df)
        df = self.standard_scaler(self,self.df)
        df = self.power_transformer(self,self.df)
        return df





class PreProcess :
    """ A class used to prepare Categorical and Numerical data that is later fed into the model  """
    # %%
    def __init__(self) -> None:
        pass
        
    # %%
    def prepare_data(self , dt:pd.DataFrame) -> pd.DataFrame: 
        """ Function to Normalize numeric Variables and OneHot encode categorical variables """
        
        dt_columns=dt.columns
        for c in dt_columns:
            print(c)
            if dt[c].dtype != "object":
                dt[c] = self.normalize(dt[c],2)
            else:
                onehot_encoded = self.onehot(dt[c], c)
                dt = dt.join(onehot_encoded)
                dt = dt.drop([c], axis=1)
        return dt


    # %%
    def normalize(self , X_skewed:pd.Series , skew_threshold:int = 2) -> pd.Series:
        """ Helper function which normalizes numeric variables using power transformation """
        
        if skew(X_skewed)>abs(skew_threshold):
            #X_Normalized, m = stats.boxcox(X_skewed)
            pt = PowerTransformer()
            X_Normalized=pt.fit_transform(X_skewed.values.reshape(-1,1))
            return X_Normalized
        else:
            return X_skewed

    # %%
    def onehot(self , X_column:pd.Series , column_name:str) -> pd.DataFrame:
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

