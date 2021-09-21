import pandas as pd
from scipy.stats import norm, skew 

from sklearn.preprocessing import PowerTransformer

from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import PowerTransformer

# import numpy as np 
# import matplotlib.pyplot as plt
# import scipy.stats as stats
# import sklearn.linear_model as linear_model
# import seaborn as sns
# import xgboost as xgb
# from scipy.special import boxcox1p
# from sklearn import preprocessing
# from scipy import stats
# from scipy import stats
# import seaborn as sns


class PreProcess :

    # %%
    def __init__(self) -> None:
        pass
        
    # %%
    def prepare_data(self , dt):
        dt_columns=dt.columns
        for c in dt_columns:
            print(c)
            if dt[c].dtype != "object":
                dt[c] = self.Normalize(dt[c],2)
            else:
                onehot_encoded = self.OneHot(dt[c], c)
                dt = dt.join(onehot_encoded)
                dt = dt.drop([c], axis=1)
        return dt


    # %%
    def Normalize(self , X_skewed, skew_threshold = 2):
        if skew(X_skewed)>abs(skew_threshold):
            #X_Normalized, m = stats.boxcox(X_skewed)
            pt = PowerTransformer()
            X_Normalized=pt.fit_transform(X_skewed.values.reshape(-1,1))
            return X_Normalized
        else:
            return X_skewed

    # %%
    def OneHot(self , X_column, column_name):
        #Labe Encoding
        lbl = LabelEncoder() 
        lbl.fit(list(X_column.values)) 
        X_labelencoded = lbl.transform(list(X_column.values))
        #One Hot encoding
        onehot_encoded=pd.get_dummies(X_labelencoded, prefix=column_name)
        #onehot_encoder = OneHotEncoder(sparse=False)
        #onehot_encoded = onehot_encoder.fit_transform(X_labelencoded.reshape(-1,1))
        return onehot_encoded

