from fastai.vision.all import *
from utils import DIR_KAGGLE , DIR_DATA

def setup_kaggle():
    file = DIR_KAGGLE.joinpath("kaggle.json")

    with open(file) as f:
        creds = json.load(f)
    
    cred_path = Path('~/.kaggle/kaggle.json').expanduser()

    if not cred_path.exists():
        cred_path.parent.mkdir(exist_ok = True)
        cred_path.mk_write(json.dumps(creds))
        cred_path.chmod(0o600)


def download_sample_datasets():
    
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    ## Competition DataSet
    TITANIC = DIR_DATA.joinpath("TITANIC")
    api.competition_download_file("titanic","train.csv",TITANIC)

    ## Loading train.csv
    filepath = TITANIC.joinpath("train.csv")
    df = pd.read_csv(filepath)
    
    ## Creating samples for testing
    sample = df.sample(30 , random_state=100)
    sample_file = TITANIC.joinpath("sample.csv")
    sample.to_csv(sample_file)