from utils.feature_engineering import FeatureScaler
from utils import DIR_SAMPLE
import pandas as pd
from tests.conftest import test_data

def test_feature_scaler(test_data):
    
    assert len(test_data) == 30


def test_feature_scaler2(test_data):
    assert 1==1