import pandas as pd
from sklearn import model_selection


if __name__ == '__main__':
    
    # reading input data
    df = pd.read_csv("~/Pet_Projects/Utils/DATA/cat-in-the-dat/train.csv")
    df['kfold'] = -1

    # Shufffling dataset using sample function
    df = df.sample(frac=1).reset_index()

    # Defining Kfold parameters from model selection 
    kf = model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    target_var = 'target'

    for fold , (train_idx , val_idx) in enumerate(kf.split(X=df , y=df[target_var].values)):
        print(len(train_idx) , len(val_idx))
        df.loc[val_idx,'kfold'] = fold
    
    df.to_csv("DATA/cat-in-the-dat/train_folds.csv")



