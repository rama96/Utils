export TRAINING_DATA=DATA/cat-in-the-dat/train_folds.csv
export FOLD=2
# export model_name='extratrees'

export model_name='extratrees'
python src/train.py 

# export model_name='randomforest'
# python src/train.py 