from typing import List
import pandas as pd
from typing import Iterable

# src/features/feature_selector.py
feature_config = { # select all features handled in preprocessing.py
    "nyc-crashes": {
        'preset-1' : ( # presets are immutable once set
            "time_from_peak_hour",
            "borough_Unspecified",
            "borough_BROOKLYN",
            "vehicle_type_code1_Sedan",
            "contributing_factor_vehicle_1_Distraction"
            # Add other relevant encoded columns as needed
        ) 
    }
}

# explicitly select or don't select one-hot-encoded names allows for more customization over only specifying the original col name and either having all one-hot-encoded columns or none of them
# remember PS171 certain onehot encoded values were way more predictive than others

def get_preset(dataset, preset='all') -> List[str]:
    """
    """
    assert dataset in feature_config.keys(), f"Inputted dataset {dataset} not supported in feature selector"
    if preset == 'all':
        return list(feature_config.get(dataset).keys())
    else:
        assert preset in feature_config.get(dataset).keys(), f"Inputted preset {preset} not supported for dataset {dataset}"
        return feature_config.get(dataset).get(preset, [])
    
def add_presets(dataset, preset_dict):
    # chat gpt to write assert statements
    assert dataset in feature_config, f"Dataset '{dataset}' not in feature_config"
    assert isinstance(preset_dict, dict), "Preset must be a dictionary"
    
    dataset_config = feature_config.get(dataset)
    for key, value in preset_dict.items():
        assert isinstance(key, str), f"Preset name '{key}' must be a string"
        assert isinstance(value, Iterable), f"Features for preset '{key}' must be iterable"
        assert all(isinstance(col, str) for col in value), "All feature names must be strings"
        assert key not in feature_config[dataset], f"Preset '{key}' already exists for dataset '{dataset}'"

        dataset_config[key] = tuple(value)

def delete_presets(dataset, presets: Iterable[str]):
    assert dataset in feature_config, f"Dataset '{dataset}' not found in feature_config"
    for preset_name in presets:
        assert preset_name in feature_config[dataset], f"Preset '{preset_name}' does not exist for dataset '{dataset}'"
        del feature_config[dataset][preset_name]

def select_features(X, dataset, preset='all') -> pd.DataFrame:
    """
    Filters columns from X based on configured list for the dataset.

    preset arg can be:
        - a string spcifying pre-initialized preset in feature_config
        - a list of col names which is a temporary preset
        - a dict with a single key value pair of <preset name> : [col names] in which case inputted preset is used and added to configuration dictionary
    """ 
    assert dataset in feature_config.keys(), f"Inputted dataset {dataset} not supported in feature selector"
    preset = preset.lower()
    if preset == 'all':
        return X.copy()
    
    if preset in feature_config.get(dataset).keys(): # preset=str => its a existing preset
        valid_features = feature_config.get(dataset).get(preset)
    elif isinstance(preset, Iterable): # preset=Iterable => temporary preset, passes in just a Iterable of cols
        assert all(elem in X.columns for elem in preset), f"Inputted column names not in design matrix with cols: {X.columns.tolist()}"
        valid_features = preset
    elif isinstance(preset, dict): # preset=dict => we're adding a new preset and running it
        assert len(preset) == 1, f"Can only input a dictionary with one preset when simultanously adding dictionary to feature selector config and running it."
        key, value = next(iter(preset.items()))
        add_presets(dataset, {key: value})
        valid_features = value
    else:
        raise ValueError(f"Invalid preset type: {preset}")
    
    # Some features in the config may not exist due to missing categories in current dataset
    selected = [col for col in valid_features if col in X.columns]
    return X[selected].copy()
