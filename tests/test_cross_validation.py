import pytest
import pandas as pd
from src.cross_validation import CrossValidation
from pandas.testing import assert_frame_equal


def test_validation_binary_classification(sample_data):
    
    # Multiple targets under binary class 
    with pytest.raises(Exception,match="Expected 1 target col for the given problem type but got many"):
        cv = CrossValidation( 
                        df = sample_data['binary_classification'].train ,
                        n_folds=5,
                        problem_type='binary_classification',
                        target_cols=['target','bin_1'],
                        shuffle=True                                                
                        )
        df = cv.split()

    
    # Only one value under same class
    with pytest.raises(Exception,match = "Only one class found in the dataframe"):
        
        df_train = sample_data['binary_classification'].train
        df_train = df_train[df_train['target'] == 0]
        cv = CrossValidation( 
                        df = df_train ,
                        n_folds=5,
                        problem_type='binary_classification',
                        target_cols=['target'],
                        shuffle=True                                                
                        )
        df = cv.split()
    
    # Stratified K-fold Cross Validation :
    cv = CrossValidation( 
                        df = sample_data['binary_classification'].train ,
                        n_folds=5,
                        problem_type='binary_classification',
                        target_cols=['target'],
                        shuffle=True                                                
                        )
    df = cv.split()
        
    # Checking len of cols
    assert len(sample_data['binary_classification'].train)==len(df)
    
    # Checking cols
    assert len(sample_data['binary_classification'].train.columns)== (len(df.columns) -1)

    # Checking difference in cols
    assert( list(set(df.columns) - set(sample_data['binary_classification'].train.columns)) == ['kfold'] )

    # Checking if shuffle worked
    df.drop(columns = ['kfold'] , inplace = True)
    assert(df.equals(sample_data['binary_classification'].train) == False)

    # Checking for Shuffle = False
    cv = CrossValidation( 
                        df = sample_data['binary_classification'].train ,
                        n_folds=5,
                        problem_type='binary_classification',
                        target_cols=['target'],
                        shuffle=False                                                
                        )
    df = cv.split()
    df.drop(columns = ['kfold'] , inplace = True)
    assert(df.equals(sample_data['binary_classification'].train) == True)





def test_validation_regression(sample_data):
    # stratified k-fold test
    assert 1==1

def test_validation_multiclassification(sample_data):
    assert 1==1

def test_validation_multilabel_classification(sample_data):
    assert 1==1

def test_validation_time_series(sample_data):
    assert 1==1


