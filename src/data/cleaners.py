# src/data/cleaners.py
import os
import pandas as pd

def load_crash_data():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data/raw/nyc-crashes.csv'))
    return pd.read_csv(path)

def consolidate_response(data):
    copy = data.copy()
    num_injured = copy['number_of_pedestrians_injured'] + copy['number_of_cyclist_injured'] + copy['number_of_motorist_injured']
    num_killed = copy['number_of_pedestrians_killed'] + copy['number_of_cyclist_killed'] + copy['number_of_motorist_killed']
    copy['Y'] = ((num_injured >= 1) | (num_killed >= 1)).astype(int)
    return copy
