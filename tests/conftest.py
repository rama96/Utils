import pytest
from src.sample_data import SampleData

@pytest.fixture
def sample_data():
    return  {
            'binary_classification' : SampleData(problem_type="binary_classification").train ,
             'multi_classification' : SampleData(problem_type="multi_classification").train , 
             'multilabel_classification' : SampleData(problem_type="multilabel_classification").train ,
             'single_col_regression' : SampleData(problem_type="regression").train ,
             'multi_col_regression' : SampleData(problem_type="regression").train ,
             'time_series' : SampleData(problem_type="time_series").train ,
            }



