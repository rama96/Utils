from sklearn import preprocessing , ensemble , metrics
import pandas as pd
import os
from src.dispatcher import MODELS
import joblib
from src import N_FOLDS , TEST_DATA , MODEL , FOLD


def predict():
    
    predictions = None
    
    for FOLD in range(N_FOLDS):
        
        
        # reading the test data
        df = pd.read_csv(TEST_DATA)

        label_encoders = joblib.load(os.path.join("models",f"models/{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models",f"models/{MODEL}_{FOLD}_columns.pkl"))
        
        # since all the variables are vategorical variables , we iterate through all columns , if not we only select categgorical variables here 
        for c in cols:
            lbl = label_encoders[c]
            df.loc[:,c] = lbl.transform(df[c].values.tolist())
        
        # loading classifiers 
        clf = joblib.load(os.path.join("models",f"models/{MODEL}_{FOLD}_model.pkl"))
        
        
        # filtering only required columns just in case there are extra columns in the test data
        df = df[cols]
        preds=clf.predict_proba(df)[:,1]
        
        # Summing all the predictions to take the average of all predictions
        if FOLD == 0 :
            
            # For the first time , predictions was assigned as None , to change that 
            predictions = preds 
        else :
            
            # Adding the predictions of the subsequent fold models 
            predictions+=preds
    
    # Taking average of all the predictions 
    predictions/=N_FOLDS

    # Adding the predictions to test_dataset
    test_data = pd.read_csv(TEST_DATA)
    test_data['predicted_proba'] = predictions

    df['predicted_proba'] = predictions

    return test_data


if __name__== "__main__":
    df_predicted = predict()
    submission = df_predicted[['ID','predicted_proba']]
    submission.columns =  ['id','target']
    
        