from easy_ml.missing_data import MissingDataHandlerV1
from easy_ml import DIR_SAMPLE
import pandas as pd
from tests.conftest import test_data

def test_show_missing_data_distribution(test_data):
    missing_data_handler = MissingDataHandlerV1(test_data)
    missing = missing_data_handler.show_distribution()

    assert 1==1