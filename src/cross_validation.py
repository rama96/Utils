from multiprocessing.sharedctypes import Value
from re import L, M
import pandas as pd
from sklearn import model_selection
from src import TRAINING_DATA
from src.sample_data import SampleData

""" 
This module is for handling crosss validation for various types of datasets :

Major problems include :

1. Binary Classification Problems 
2. Multi Class Classification Problems 
3. Multi Label Classification Problems 
4. Single Column regression Problems 
5. Multi Column Classification Problems 
6. Holdout

"""

class CrossValidation:
    def __init__(self , 
                df , 
                target_cols , 
                n_folds = 5 , 
                problem_type = "binary_classification", 
                shuffle = True ,
                multilabel_delimiter = ","
                ):
        """  A class that returns the train and cv splits of the data depending on the problem type and other parameters

        Parameters :
        ----------
            - df : pd.DataFrame 
            The df for which you need to perform the splits 
            
            - n_folds : int 
            The number of folds of CV you need

            - problem_type : str    
            Accepts 5 problem types for now :
                            
                            1. 'binary_classification'
                            2. 'multi_classification'
                            3. 'multilabel_classification'
                            4. 'single_col_regression'
                            5. 'multi_col_regression'
                            6. 'holdout_x'
            
            - target_cols : list 
            The list of independent varaibles ,  ex : ['target']

            - shuffle : bool 
            A binary value to check if the dataframe needs to be shuffled .

            - multilabel_delimiter : str
            A delimiter used in case of multilabel classification to seperate out the classes
        ----------
        
        """
        self.df = df.copy()
        self.n_folds = n_folds
        self.target_cols = target_cols
        self.problem_type = problem_type
        self.shuffle = shuffle
        self.num_targets = len(self.target_cols)
        self.multilabel_delimiter = multilabel_delimiter
    
    def split(self):
        
        # Shuffling the dataframe if not done already
        if self.shuffle == True:
            self.df = self.df.sample(frac=1).reset_index(drop = True)

        # Handling classification problems ( 1 and 2 ) - Use stratified K-Fold cross validation
        if self.problem_type in ['binary_classification','multi_classification']:
        
            # Single target col Exception
            if self.num_targets > 1:
                raise Exception("Expected 1 target col for the given problem type but got many")
                
            target_col = self.target_cols[0]
            n_classes = self.df[target_col].nunique()
            
            if n_classes == 1:
                raise Exception("Only one class found in the dataframe")
            
            # Stratified K-Fold cross validation
            self.df['kfold'] = -1
            kf = model_selection.StratifiedKFold(n_splits=self.n_folds,shuffle = False)
            for fold , (train_idx , val_idx) in enumerate(kf.split(X=self.df , y=self.df[target_col].values)):
                self.df.loc[val_idx,'kfold'] = fold
        
    
        # Handling Regression problems (4 , 5) - Use normal K-fold cross validation
        # TODO: Better way to implement Straified K-Fold here 

        elif self.problem_type in ['single_col_regression','multi_col_regression'] :
            
            # Handling exceptions
            if (self.problem_type == 'single_col_regression') & (self.num_targets > 1) :
                raise Exception("Problem-type of single_col_regression expected 1 target_col but got multiple cols")
            if (self.problem_type == 'multi_col_regression') & (self.num_targets == 1) :
                raise Exception("Problem-type of multi_col_regression expected multiple target_col but got 1 col")
            
            # K-fold
            kf = model_selection.KFold(n_splits=self.n_folds, shuffle=False)
            for fold , (train_idx , val_idx) in enumerate(kf.split(X=self.df , y=self.df[self.target_cols].values)):
                self.df.loc[val_idx,'kfold'] = fold
        
        # 6. Holdout mainly used in time series models - Saving some future data as samples to test out our cross validation 
        elif self.problem_type.startswith('holdout_') :
            
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.df) * holdout_percentage / 100)
            
            # Seperating data by the index by holding out a certain percentage of data
            self.df.loc[:len(self.df) - num_holdout_samples, "kfold"] = 0
            self.df.loc[len(self.df) - num_holdout_samples:, "kfold"] = 1

        
        # 3. Follow stratified K-fold on the list of targets 
        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Expected targets to be comma seperated in a single col but got list of targets")
            target_col = self.target_cols[0]
            targets = self.df[target_col].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.n_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.df, y=targets)):
                self.df.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception("Invalid Problem type recieved")

        return self.df
    
                
if __name__ == '__main__':
    
    df = SampleData("regression").train
    cv = CrossValidation(df = df , target_cols=['SalePrice'], n_folds=5 , problem_type = "single_col_regression",shuffle=True)
    cv_data = cv.split()
    print(cv_data.head())
    print(cv_data.kfold.value_counts())
