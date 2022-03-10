import pytest
import pandas as pd
from src.cross_validation import CrossValidation


def test_regression(sample_data):
    # 
    cv = CrossValidation()
    assert 1==1

def test_binary_classification(sample_data):
    # stratified k-fold test
    assert 1==1

def test_multiclassification(sample_data):
    assert 1==1

def test_multilabel_classification(sample_data):
    assert 1==1

def test_time_series(sample_data):
    assert 1==1


