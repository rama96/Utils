from sklearn import preprocessing
from src.sample_data import SampleData
import pandas as pd

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        # Resting index to aviod errors
        self.df.reset_index(drop = True , inplace = True)
        
        # Replacing NAs with -9999999
        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        
        # Taking deep copy and not shallow copy
        self.output_df = self.df.copy(deep=True)
    
    #TODO : frequency based encoding 
    
    
    #TODO : target based encoding 
    
    
    # Helper function for label encoding using sklearn label encoder. Nothing fancy
    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    
    # Binary Labels ex : 100 010 001 using sklearn preprocessor
    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    # One hot encoding using Sklearn preprocessor
    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats].values)

    # One hot encoding using pd.dummies
    def _one_hot_2(self):
        for col in self.cat_feats:
            onehot_encoded=pd.get_dummies(self.df[col].values, prefix=col)
            # removing the last redundant variable 
            onehot_encoded = onehot_encoded.iloc[:, :-1]
            self.output_df = pd.concat([self.output_df, onehot_encoded], axis = 1)

        self.output_df = self.output_df.drop(columns=self.cat_feats , axis = 1)
        return self.output_df

    
    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        elif self.enc_type == "ohe2":
            return self._one_hot_2()
        else:
            raise Exception("Encoding type not understood")
    
    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                
                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        elif self.enc_type == "ohe":
            return self.ohe(dataframe[self.cat_feats].values)
        
        else:
            raise Exception("Encoding type not understood")
                

if __name__ == "__main__":
    import pandas as pd
    from sklearn import linear_model
    from src import DIR_DEBUG
    import numpy as np
    
    binary_sample = SampleData("binary_classification")

    df = binary_sample.train
    df_test = binary_sample.test
    # sample = pd.read_csv("../input/sample_submission.csv")

    train_len = len(df)

    df_test["target"] = -1
    full_data = pd.concat([df, df_test])

    cols = [c for c in df.columns if c not in ["id", "target"]]
    cat_feats = CategoricalFeatures(full_data, 
                                    categorical_features=cols, 
                                    encoding_type="ohe2",
                                    handle_na=True)
    full_data_transformed = cat_feats.fit_transform()
    
    full_data_transformed = full_data_transformed.drop(columns = 'id' , axis = 1)
    X = full_data_transformed[:train_len].values
    X_test = full_data_transformed[train_len:].values

    clf = linear_model.LogisticRegression()
    clf.fit(X, df.target.values)
    preds = clf.predict_proba(X_test)[:, 1]
    
    print(full_data_transformed.head())
    full_data_transformed['predicted_probabilities'] = -1
    full_data_transformed.loc[train_len:,'predicted_probabilities'] = np.round(preds,2)
    
    filepath = DIR_DEBUG.joinpath("categorical_results.csv")

    full_data_transformed.to_csv(filepath)