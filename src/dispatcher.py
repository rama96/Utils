from sklearn import ensemble

MODEL = {

    'randomforest' : ensemble.RandomForestClassifier(n_estimators = 300 ,  n_jobs=-1,verbose = 2),
    'extratrees' : ensemble.ExtraTreesClassifier(n_estimators = 300 ,  n_jobs=-1,verbose = 2),

}