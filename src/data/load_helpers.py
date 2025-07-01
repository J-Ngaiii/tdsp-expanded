import pandas as pd
import os

from src.config.config import get_output_path

def download_csv(name, data, rtrn=False):
    df = pd.DataFrame(data)
    if rtrn:
        return df
    else:
        path = get_output_path(name, 'raw') # CIRCULAR DEPENENCY
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', path))
        os.makedirs(output_dir, exist_ok=True)  # ensure the output dir exists

        output_path = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved {len(data)} records to {output_path}")

DOWNLOADERS = {
    'csv' : download_csv
}

def get_downloader(key):
    return DOWNLOADERS.get(key)