from easy_ml import DIR_SAMPLE
import pandas as pd
import pytest 

@pytest.fixture
def test_data():
    filepath = DIR_SAMPLE.joinpath("sample.csv")
    df = pd.read_csv(filepath)
    return df
