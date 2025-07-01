from typing import Set

from src.data.load_helpers import download_csv

DATA_URL = {
    'nyc motor' : {
        'url' : "https://data.cityofnewyork.us/resource/bm4k-52h4.json", 
        'params' : {
            '$select': "*", 
            '$where': "crash_date >= '2015-01-01T00:00:00.000'", # get last 10 years worth of data
            '$limit': 4000000
            }, 
        'rettype' : download_csv, 
        'paths' : {'raw' : 'data/raw', 'processed' : 'data/processed', 'interim' : 'data/interim'}
    }
}

def get_data_master(name, field):
    assert isinstance(name, str), f"Dataset name is not a string but type {type(name)}."
    assert isinstance(field, str), f"field is not a string but type {type(field)}."

    return DATA_URL.get(name).get(field, "")

def get_data_url(name='nyc motor'):
    assert isinstance(name, str), f"Dataset name  is not a string but type {type(name)}."

    return DATA_URL.get(name).get('url', "")

def get_all__dataset_names() -> Set[str]:
    return set(DATA_URL.keys())

def get_output_path(name, pathtype):
    assert isinstance(name, str), f"Dataset name is not a string but type {type(name)}."
    assert isinstance(pathtype, str), f"Inputted pathtype is not a string but type {type(pathtype)}."

    return DATA_URL.get(name).get('paths').get(pathtype, "")