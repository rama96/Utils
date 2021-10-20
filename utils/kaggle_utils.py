from fastai.vision.all import *
from utils import DIR_KAGGLE , DIR_DATA , create_directory_if_not_exists
file = DIR_KAGGLE.joinpath("kaggle.json")
import pathlib


def setup_kaggle(filepath:typing.Union[str, pathlib.Path] = file):
    """ 
        Reads the downlaoded kaggle.json from the specified filepath from
        and stores it ~/.kaggle folder to setup kaggle library

    Parameters
	----------
	filepath : typing.Union[str, pathlib.Path]
		Accepts the path of kaggle.json file
	Returns
	-------
	None

    """
    
    with open(filepath) as f:
        creds = json.load(f)
    
    cred_path = Path('~/.kaggle/kaggle.json').expanduser()

    if not cred_path.exists():
        cred_path.parent.mkdir(exist_ok = True)
        cred_path.mk_write(json.dumps(creds))
        cred_path.chmod(0o600)
    

def download_sample_datasets():
    """ 
        Downloads train.csv from titanic datasets and creates a sample dataset with 25 odd entries for testing purposes under the folder SAMPLE inside DATA
    Parameters
	----------
	None
    
    Returns
    -------
    None
	
    """
    
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    ## Competition DataSet
    TITANIC = DIR_DATA.joinpath("TITANIC")
    api.competition_download_file("titanic","train.csv",TITANIC)

    ## Loading train.csv
    filepath = TITANIC.joinpath("train.csv")
    df = pd.read_csv(filepath)
    
    ## Creating Samples
    DIR_SAMPLE = DIR_DATA.joinpath('SAMPLE')
    create_directory_if_not_exists(DIR_SAMPLE)
    
    ## Creating samples for testing
    sample = df.sample(30 , random_state=100)
    sample_file = DIR_SAMPLE.joinpath("sample.csv")
    sample.to_csv(sample_file)