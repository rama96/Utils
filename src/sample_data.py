import pandas as pd
from src import DIR_SAMPLES

class SampleData:
    def __init__(self,problem_type):
        self.problem_type = problem_type
        
        if problem_type == "binary_classification":
            DIR_DATASET = DIR_SAMPLES.joinpath("binary_classification")
            
        elif problem_type == "multi_classification":
            DIR_DATASET = DIR_SAMPLES.joinpath("multi_classification")
            
        elif problem_type == "multilabel_classification":
            DIR_DATASET = DIR_SAMPLES.joinpath("multilabel_classification")
            
        elif problem_type == "regression":
            DIR_DATASET = DIR_SAMPLES.joinpath("regression")
            
        elif problem_type == "time_series":
            DIR_DATASET = DIR_SAMPLES.joinpath("time_series")
            
        else :
            raise Exception("Invalid problem type entered")

        self.test = pd.read_csv(DIR_DATASET.joinpath("train.csv"))
        self.train = pd.read_csv(DIR_DATASET.joinpath("test.csv"))
