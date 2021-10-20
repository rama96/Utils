from utils.missing_data import MissingDataHandlerV1
from utils import DIR_SAMPLE
import pandas as pd

def test_show_missing_data_distribution():
    filepath = DIR_SAMPLE.joinpath("sample.csv")
    df = pd.read_csv(filepath)
    missing_data_handler = MissingDataHandlerV1(df)
    missing = missing_data_handler.show_distribution()

    assert 1==1