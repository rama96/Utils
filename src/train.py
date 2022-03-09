from sklearn import preprocessing , ensemble , metrics
import pandas as pd
import os
from src.dispatcher import MODELS
import joblib
from src import MODEL , TRAINING_DATA , N_FOLDS , FOLD , TEST_DATA

print(MODEL)

target_var = 'target'

# relevant_cols = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
# categorical_cols = ['Sex','Cabin']

FOLD_MAPPING = {}
for i in range(N_FOLDS):
    FOLD_MAPPING[i] = [j for j in range(N_FOLDS) if i!=j]


if __name__== "__main__":
    
    # reading the training data
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    
    # dividing df into train and valid based on fold values
    train_df = df[df['kfold'].isin(FOLD_MAPPING[FOLD])]
    valid_df = df[~df['kfold'].isin(FOLD_MAPPING[FOLD])]

    # Converting y series to array
    ytrain = train_df[target_var].values
    yvalid = valid_df[target_var].values

    # Dropping Unnecessary columns
    vars_to_be_dropped = ['id','kfold',target_var]
    train_df = train_df.drop(vars_to_be_dropped,axis=1)
    valid_df = valid_df.drop(vars_to_be_dropped,axis=1)

    # Making sure valid and train contain the column order
    valid_df = valid_df[train_df.columns]
    
    # TODO:  missing values to be handled seperately but not right now

    label_encoders = {}
    
    # since all the variables are vategorical variables , we iterate through all columns , if not we only select categgorical variables here 
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + df_test[c].values.tolist())
        train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    # Data is ready to be trained 
    clf = MODELS[MODEL]
    clf.fit(train_df,ytrain)
    preds=clf.predict_proba(valid_df)[:,1]
    
    print(metrics.roc_auc_score(yvalid ,preds))

    joblib.dump(label_encoders , f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf , f"models/{MODEL}_{FOLD}_model.pkl")
    joblib.dump(train_df.columns , f"models/{MODEL}_{FOLD}_columns.pkl")