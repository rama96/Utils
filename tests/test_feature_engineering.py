from easy_ml.feature_engineering import FeatureScaler , CategoricalEncoder
from easy_ml import DIR_SAMPLE
import pandas as pd
from tests.conftest import test_data

def test_feature_scaler(test_data):
    assert len(test_data) == 30
    
    min_max_cols = ['PassengerId','Fare']
    standard_cols = ['Age']
    
    feature_scaler = FeatureScaler(min_max_cols=min_max_cols,standard_cols=standard_cols)
    test_data = feature_scaler.fit_transform(test_data)

    assert len(test_data) == 30


def test_categorical_encoder(test_data):
    
    assert test_data.shape == (30,13)
    nominal_cols = ['Pclass','Sex']
    enc = CategoricalEncoder(nominal_cols=nominal_cols)
    df_transformed = enc.fit_transform(test_data)
    assert df_transformed.shape == (30,14)



# df= pd.DataFrame({'col1':['a','b','c','a','d'],'col2':['a','c','b','d','e']})