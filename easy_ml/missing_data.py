""" Module to handle Missing values in the data """

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.neighbors import KNeighborsRegressor

class MissingDataHandlerV1:
    """ A class that takes in the input as pd.DataFrame to analyze missing data and impute Numerical and Categorical values 
    TODO : 1. Accept a argument use_default_categorization = True which automatically categorized numerical and categorical values and stores it as attributes
    """
    
    def __init__(self , df:pd.DataFrame = None ) -> None:
        self.df = df
        self.show_distribution()
        self.assign_default_data_labels()
    
    def show_distribution(self) -> None:
        """ Shows Distribution of Missing data in the dataframe inputed """
        
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        missing = missing/len(self.df)
        missing.sort_values(ascending = False, inplace=True)
        missing = missing * 100
        # print(missing.head(50))
        # missing.plot.bar()
        return missing
        

    def assign_default_data_labels(self) -> None:
        """ Assigns data labels to categorical and numerical variables """
        
        self.numeric_data = self.df.select_dtypes(include=[np.number])
        self.categorical_data = self.df.select_dtypes(exclude=[np.number])
        print("Numerical Data : " , self.numeric_data.head())
        print("Categorical Data : " , self.categorical_data.head())

    def numerical_impute(self) -> None:
        """ Imputes valies for DataFrame containing numerical columns """
        
        my_imputer = SimpleImputer()
        _df = self.numeric_data.copy()
        _df = pd.DataFrame(my_imputer.fit_transform(_df))
        _df.columns = self.df.columns
        _df.index = self.df.index
        missing_new = _df.isnull().sum()
        print(missing_new)
        self.numeric_imputed_data = _df

    # %%
    def categorical_impute(self) -> None :
        """ Imputes valies for DataFrame containing Categorical columns """
        #instantiate both packages to use
        encoder = OrdinalEncoder()
        imputer = IterativeImputer(ExtraTreesRegressor())
        #create a list of categorical columns to iterate over
        cat_cols = self.categorical_data.columns

        def encode(data):
            '''function to encode non-null data and replace it in the original data'''
            #retains only non-null values
            nonulls = np.array(data.dropna())
            #reshapes the data for encoding
            impute_reshape = nonulls.reshape(-1,1)
            #encode date
            impute_ordinal = encoder.fit_transform(impute_reshape)
            #Assign back encoded values to non-null values
            data.loc[data.notnull()] = np.squeeze(impute_ordinal)
            return data

        #create a for loop to iterate through each column in the data
        for columns in cat_cols:
            encode(self.categorical_data[columns])
        encode_data = pd.DataFrame(np.round(imputer.fit_transform(self.categorical_data)),columns = self.categorical_data.columns)
        self.categorical_imputed_data = encode_data
        
    
if __name__ == "main":
    pass