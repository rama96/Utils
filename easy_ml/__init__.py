import warnings
from pathlib import Path
import os
import pandas as pd
from datetime import datetime 

# from easy_ml.feature_engineering import *
# from easy_ml.missing_data import *
# from easy_ml.models import *
# from easy_ml.viz import *

today_time = datetime.today()

TODAY = today_time.strftime('%Y-%m-%d')
TODAY_SUFFIX = (datetime.today()).strftime('%Y%m%d')

warnings.simplefilter(action="ignore", category=FutureWarning)

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# your performance may suffer as PyTables will pickle object types that it cannot


env = os.getenv("PROMOTIONS_ENV", "production").lower()
if env.lower() != "production":
    print(f"\n***** ENVIROMENT: {env.upper()} *****")


def create_directory_if_not_exists(path: Path) -> None:
    if not path.is_dir():
        try:
            path.mkdir()
        except Exception as e:
            print(e)


DIR_PACKAGE = Path(__file__).resolve().parent  # ../easy_ml/easy_ml
DIR_BASE = DIR_PACKAGE.parent  # ../easy_ml/

# ../easy_ml/DATA
DIR_DATA = DIR_BASE.joinpath("DATA")

# ../easy_ml/DATA
DIR_KAGGLE = DIR_BASE.joinpath("kaggle")

# ../easy_ml/SAMPLE
DIR_SAMPLE = DIR_DATA.joinpath("SAMPLE")

create_directory_if_not_exists(DIR_DATA)
create_directory_if_not_exists(DIR_SAMPLE)

name = "easy_ml"

pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 25)
pd.options.display.float_format = "{:.2f}".format
pd.options.mode.chained_assignment = None




