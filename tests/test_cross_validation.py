from tkinter.tix import NoteBook
import pytest
import pandas as pd
from src.cross_validation import CrossValidation
from pandas.testing import assert_frame_equal
import math
    

class TestCrossValidation:
    
    def _perform_standard_checks(self, train_data ,cv_data , shuffle_check):
        
        # Checking len of cols
        assert len(train_data)==len(cv_data)
        
        # Checking cols
        assert len(train_data.columns)== (len(cv_data.columns) -1)

        # Checking difference in cols
        assert( list(set(cv_data.columns) - set(train_data.columns)) == ['kfold'] )

        # Checking if shuffle worked
        cv_data.drop(columns = ['kfold'] , inplace = True)
        assert(cv_data.equals(train_data) == shuffle_check)

    
    def test_validation_binary_classification(self,sample_data):
    
        train_data = sample_data['binary_classification']
        # Multiple targets under binary class 
        with pytest.raises(Exception,match="Expected 1 target col for the given problem type but got many"):
            cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='binary_classification',
                            target_cols=['target','bin_1'],
                            shuffle=True                                                
                            )
            df = cv.split()

        
        # Only one value under same class
        with pytest.raises(Exception,match = "Only one class found in the dataframe"):
            
            df_single_class = train_data[train_data['target'] == 0]
            cv = CrossValidation( 
                            df = df_single_class ,
                            n_folds=5,
                            problem_type='binary_classification',
                            target_cols=['target'],
                            shuffle=True
                            )
            df = cv.split()
        
        # Stratified K-fold Cross Validation :
        cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='binary_classification',
                            target_cols=['target'],
                            shuffle=True
                            )
        cv_data = cv.split()

        self._perform_standard_checks(train_data , cv_data , shuffle_check = False)  

        
        # Checking for Shuffle = False
        cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='binary_classification',
                            target_cols=['target'],
                            shuffle=False
                            )
        cv_data = cv.split()
        
        self._perform_standard_checks(train_data , cv_data , shuffle_check = True)  

    def test_validation_single_col_regression(self,sample_data):
        
        train_data = sample_data['single_col_regression']
        # Single Col regression with multiple targets
        with pytest.raises(Exception,match="Problem-type of single_col_regression expected 1 target_col but got multiple cols"):
            cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='single_col_regression',
                            target_cols=['SalePrice','LotArea'],
                            shuffle=True
                            )
            df = cv.split()

        # multi Col regression with single target
        with pytest.raises(Exception,match="Problem-type of multi_col_regression expected multiple target_col but got 1 col"):
            cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='multi_col_regression',
                            target_cols=['SalePrice'],
                            shuffle=True
                            )
            df = cv.split()

        
        # Single col regression with single target
        cv = CrossValidation( 
                        df = train_data ,
                        n_folds=5,
                        problem_type='single_col_regression',
                        target_cols=['SalePrice'],
                        shuffle=True
                        )
        cv_data = cv.split()
        
        self._perform_standard_checks(train_data , cv_data , shuffle_check = False)
        
    def test_validation_multi_col_regression(self,sample_data):
        
        train_data = sample_data['multi_col_regression']
        
        # Mutlti Col regression with multiple targets
        cv = CrossValidation( 
                df = train_data ,
                n_folds=5,
                problem_type='multi_col_regression',
                target_cols=['SalePrice','LotArea'],
                shuffle=True
                )
        cv_data = cv.split()
        
        self._perform_standard_checks(train_data , cv_data , shuffle_check = False)
        
        assert 1==1


    def test_validation_multi_col_regression(self,sample_data):
        
        train_data = sample_data['multi_col_regression']
        # Multiple targets under binary class 
        with pytest.raises(Exception,match="Problem-type of multi_col_regression expected multiple target_col but got 1 col"):
            cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='multi_col_regression',
                            target_cols=['SalePrice'],
                            shuffle=True                                                
                            )
            df = cv.split()

        
        # Stratified K-fold Cross Validation :
        cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='multi_col_regression',
                            target_cols=['SalePrice','LotArea'],
                            shuffle=True
                            )
        cv_data = cv.split()

        self._perform_standard_checks(train_data , cv_data , shuffle_check = False)
        
        assert 1==1

    def test_validation_multiclassification(self,sample_data):
        
        train_data = sample_data['multi_classification']
        # Multiple targets under binary class 
        with pytest.raises(Exception,match="Expected 1 target col for the given problem type but got many"):
            cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='multi_classification',
                            target_cols=['label','petal_length'],
                            shuffle=True                                                
                            )
            df = cv.split()

        
        # Only one value under same class
        with pytest.raises(Exception,match = "Only one class found in the dataframe"):
            
            df_single_class = train_data[train_data['label'] == 'Iris-setosa']
            cv = CrossValidation( 
                            df = df_single_class ,
                            n_folds=5,
                            problem_type='multi_classification',
                            target_cols=['label'],
                            shuffle=True
                            )
            df = cv.split()
        
        # Stratified K-fold Cross Validation :
        cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='multi_classification',
                            target_cols=['label'],
                            shuffle=True
                            )
        cv_data = cv.split()

        self._perform_standard_checks(train_data , cv_data , shuffle_check = False)  

        
        # Checking for Shuffle = False
        cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='multi_classification',
                            target_cols=['label'],
                            shuffle=False
                            )
        cv_data = cv.split()
        
        self._perform_standard_checks(train_data , cv_data , shuffle_check = True)  


    def test_validation_multilabel_classification(self,sample_data):
        train_data = sample_data['multilabel_classification']

        # Only one value under same class
        with pytest.raises(Exception,match = "Expected targets to be comma seperated in a single col but got list of targets"):
        
            # Checking for Shuffle = False
            cv = CrossValidation( 
                                df = train_data ,
                                n_folds=5,
                                problem_type='multilabel_classification',
                                target_cols=['attribute_ids','id'],
                                shuffle=False,
                                multilabel_delimiter=" "
                                )
            df = cv.split()
        
        
        # Checking for Shuffle = False
        cv = CrossValidation( 
                            df = train_data ,
                            n_folds=3,
                            problem_type='multilabel_classification',
                            target_cols=['attribute_ids'],
                            shuffle=False,
                            multilabel_delimiter=" "
                            )
        cv_data = cv.split()

        self._perform_standard_checks(train_data , cv_data , shuffle_check = True)
        
            

    def test_validation_time_series(self,sample_data):
        
        train_data = sample_data['time_series']
        # Checking for Shuffle = False
        cv = CrossValidation( 
                            df = train_data ,
                            n_folds=5,
                            problem_type='holdout_30',
                            target_cols=['attribute_ids'],
                            shuffle=False,
                            multilabel_delimiter=" "
                            )
        cv_data = cv.split()

        self._perform_standard_checks(train_data , cv_data , shuffle_check = True)
        








# Checking len k-fold
        # kfold_dist = df.groupby(['kfold']).size().reset_index(drop = True)
        # kfold_dist = math.floor(kfold_dist)
        # unique_folds = list(df['kfold'].unique())
        # for i in unique_folds:
        #     fold_size = len(df[df['kfold']==i])
        #     mask = fold_size >= math.floor(len(df)/)
        #     assert()
        
        # # Checking if shuffle worked
        # df.drop(columns = ['kfold'] , inplace = True)
        # assert(df.equals(sample_data['single_col_regression'].train) == False)





# def standard_validation(df_valid , df_train):
#     assert 1==2






