# Production Use : 

```bash
pip install -e git+https://github.com/rama96/Utils.git#egg=easy_ml
pip install git+https://github.com/rama96/Utils.git

```

# Development easy_ml

A package which can be used for preparation of data , importing models , get predictions etc for personal use .  

## For Developement

```bash
virtualenv -p python3 env
source env/bin/activate
printf "\n# Adding this command to read local .env file" >> env/bin/activate
printf "\nexport \$(grep -v '^#' .env | xargs)" >> env/bin/activate
```

## For Testing

```bash
virtualenv -p python3 production_env
source production_env/bin/activate
pip install -e git+https://github.com/rama96/Utils.git#egg=easy_ml
```

```python
import easy_ml
import easy_ml
```