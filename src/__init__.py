import os

RAW_DATA = os.environ.get("RAW_DATA")
TRAINING_DATA = os.environ.get("TRAINING_DATA")
MODEL = os.environ.get("MODEL")
N_FOLDS = int(os.environ.get("N_FOLDS"))
FOLD = int(os.environ.get("FOLD"))
TEST_DATA = os.environ.get("TEST_DATA")

