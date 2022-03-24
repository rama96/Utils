import os
from pathlib import Path


# Loading Environment Variables

RAW_DATA = os.environ.get("RAW_DATA")
TRAINING_DATA = os.environ.get("TRAINING_DATA")
MODEL = os.environ.get("MODEL")
N_FOLDS = int(os.environ.get("N_FOLDS"))
FOLD = int(os.environ.get("FOLD"))
TEST_DATA = os.environ.get("TEST_DATA")
SUBMISSION_DATA = os.environ.get("SUBMISSION_DATA")



# Path varibales

DIR_PACKAGE = Path(__file__).resolve().parent  # ../promotions/promotions
DIR_BASE = DIR_PACKAGE.parent  # ../promotions/

# ../DATA
DIR_DATA = DIR_BASE.joinpath("DATA")

# ../DATA/DEBUG
DIR_DEBUG = DIR_DATA.joinpath("DEBUG")

# ../DATA
DIR_MODELS = DIR_BASE.joinpath("models")

# ../samples
DIR_SAMPLES = DIR_BASE.joinpath("samples")



