import pytest
from src.sample_data import SampleData

@pytest.fixture
def sample_data():
    return  {
            'binary_classification' : SampleData(problem_type="binary_classification") ,
             'multi_classification' : SampleData(problem_type="multi_classification") , 
             'multilabel_classification' : SampleData(problem_type="multilabel_classification") ,
             'regression' : SampleData(problem_type="regression") ,
             'time_series' : SampleData(problem_type="time_series") ,
            }

