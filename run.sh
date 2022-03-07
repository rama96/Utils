export TRAINING_DATA=DATA/cat-in-the-dat/train_folds.csv
export N_FOLDS=5
export MODEL='randomforest'

for ((i=0;i<N_FOLDS;i++));
do
   echo "FOLD-"$i" Running"
   FOLD=$i python src/train.py 
done


