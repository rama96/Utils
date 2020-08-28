import pandas as pd
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew 
from scipy.special import boxcox1p
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer

from scipy.stats import skew
from scipy import stats
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import PowerTransformer

def missing_data_distribution(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing = missing/len(df)
    missing.sort_values(ascending = False, inplace=True)
    missing = missing * 100
    missing.plot.bar()
    print(missing.head(20))


def seperating_data(df):
    numeric_data = df.select_dtypes(include=[np.number])
    categorical_data = df.select_dtypes(exclude=[np.number])
    print(numeric_data.head())
    print(categorical_data.head())
    return numeric_data,categorical_data
#categorical_data.head()

def numerical_impute(df):
    my_imputer = SimpleImputer()
    new_data_1 = df.copy()
    my_imputer = SimpleImputer()
    new_data_1 = pd.DataFrame(my_imputer.fit_transform(new_data_1))
    new_data_1.columns = df.columns
    new_data_1.index = df.index
    missing_new = new_data_1.isnull().sum()
    print(missing_new)
    return new_data_1


# %%
def categorical_impute(df):
    #instantiate both packages to use
    encoder = OrdinalEncoder()
    imputer = IterativeImputer(ExtraTreesRegressor())
    #create a list of categorical columns to iterate over
    cat_cols = df.columns

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
        encode(df[columns])
    encode_data = pd.DataFrame(np.round(imputer.fit_transform(df)),columns = df.columns)
    return encode_data


# %%
def numericorcategorical(dt):
    dt_columns=dt.columns
    for c in dt_columns:
        print(c)
        if dt[c].dtype != "object":
            dt[c] = Normalize(dt[c],2)
        else:
            onehot_encoded = OneHot(dt[c], c)
            dt = dt.join(onehot_encoded)
            dt = dt.drop([c], axis=1)
    return dt


# %%
def Normalize(X_skewed, skew_threshold = 2):
    if skew(X_skewed)>abs(skew_threshold):
        #X_Normalized, m = stats.boxcox(X_skewed)
        pt = PowerTransformer()
        X_Normalized=pt.fit_transform(X_skewed.values.reshape(-1,1))
        return X_Normalized
    else:
        return X_skewed


def OneHot(X_column, column_name):
    #Labe Encoding
    lbl = LabelEncoder() 
    lbl.fit(list(X_column.values)) 
    X_labelencoded = lbl.transform(list(X_column.values))
    #One Hot encoding
    onehot_encoded=pd.get_dummies(X_labelencoded, prefix=column_name)
    #onehot_encoder = OneHotEncoder(sparse=False)
    #onehot_encoded = onehot_encoder.fit_transform(X_labelencoded.reshape(-1,1))
    return onehot_encoded

