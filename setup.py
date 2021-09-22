import pathlib
from setuptools import setup, find_packages
import os

def _read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except IOError:
        return ""

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'easy_utils'
AUTHOR = 'Ramamurthi'
AUTHOR_EMAIL = 'ramamurthi96@gmail.com'
URL = 'https://github.com/rama96/Utils.git'

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'Utils for Data Processing and Machine Learning'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    l
    for l in _read("requirements.txt").split("\n")
    if l and not l.startswith("#") and not l.startswith("git")
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )