import requests
import time
from typing import LiteralString
import pandas as pd

from src.config.config import get_all__dataset_names, get_data_master
from src.data.load_helpers import download_csv

def load_data(name, fetchall=False, params=None, rtrn=False):
    URL = get_data_master(name, 'url')

    if all:
        URL = get_data_master(name, 'url')
        LIMIT = 50000
        offset = 0
        data = []
        while True:
            print(f"Fetching rows {offset} to {offset + LIMIT - 1}...")

            params = {
                "$limit": LIMIT,
                "$offset": offset
            }

            response = requests.get(URL, params=params)

            if response.status_code != 200:
                print(f"Request failed at offset {offset}: {response.status_code}")
                break

            batch = response.json()

            if not batch:
                print("No more data available.")
                break

            data.extend(batch)
            offset += LIMIT
            time.sleep(1)  # Respectful pause to avoid rate-limiting
        
        print(f"Total rows fetched: {len(data)}")
        if rtrn:
            return download_csv(name, data, rtrn=True) # loads data into current working directory
        else:
            download_csv(name, data, rtrn=False)
    else:
        if params is None:
            params = get_data_master(name, 'params')
            if not params:
                raise ValueError(f"No built in parameters for dataset {name}, please define parameters")
        response = requests.get(URL, params=params)

        if response.status_code == 200:
            data = response.json()
            print(f"Total rows fetched: {len(data)}")
            if rtrn:
                return download_csv(name, data, rtrn=True) # loads data into current working directory
            else:
                download_csv(name, data, rtrn=False)
        else:
            print("Request failed:", response.status_code)

def query(name: LiteralString, select: str = "*", where: str = "1=1", limit: int = 1000, write: bool = True) -> pd.DataFrame:
    """Uses SoQL"""
    assert isinstance(name, str), f"dataset_name arg must be a string but is type {type(name)}"
    assert name in get_all__dataset_names(), f"dataset_name not found, existing dataset names include {get_all__dataset_names()}"

    assert isinstance(select, str), f"select arg must be a string but is type {type(select)}"
    assert isinstance(where, str), f"where arg must be a string but is type {type(where)}"
    assert isinstance(limit, int), f"limit arg must be a int but is type {type(limit)}"
    assert isinstance(write, bool), f"write arg must be a booleam but is type {type(write)}"

    params = {
        "$select": select,
        "$where": where,
        "$limit": limit
    }

    df = load_data(name=name, fetchall=False, params=params, rtrn=True)
    return df


