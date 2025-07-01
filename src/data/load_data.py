import pandas as pd

from src.data.loaders import load_data 

if __name__ == '__main__':
    ALL = True
    DEFAULT = True
    CUSTOM = False
    param_settings = [
        {'$select':"", 
         '$where':"", 
         '$limit': 1000}
    ]
    
    NAME = 'nyc-people'
    
    if ALL:
        load_data(name=NAME, fetchall=True, params=None)
    elif DEFAULT:
        load_data(name=NAME, fetchall=False, params=None)
    elif CUSTOM:
        param = param_settings[0]
        load_data(name=NAME, fetchall=False, params=param)  
    else:
        print("No data loaded")