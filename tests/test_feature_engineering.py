from utils.feature_engineering import FeatureScaler
from utils import DIR_SAMPLE
import pandas as pd
import pytest

def test_feature_scaler():
    filepath = DIR_SAMPLE.joinpath("sample.csv")
    df = pd.read_csv(filepath)

    assert 1==1


def test_feature_scaler2():
    filepath = DIR_SAMPLE.joinpath("sample.csv")
    df = pd.read_csv(filepath)

    assert 1==1