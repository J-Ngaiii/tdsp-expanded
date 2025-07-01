from src.data.cleaners import load_crash_data, consolidate_response
from src.features.build_features import CrashFeatureBuilder
from src.features.feature_selector import select_features, get_preset

import pandas as pd
from typing import Tuple

class Preprocessing():
    def __init__(self, dataset, pipetype):
        self.feature_builder = CrashFeatureBuilder()

        self.config = {
            "nyc-crashes": {
                "response_pipe": self.CRASH_response_pipe,
                "feature_pipe": self.CRASH_feature_pipe,
                "full_pipe": self.CRASH_full_pipe,
                "features": 'all'
            }
        }

        assert dataset in self.config
        assert pipetype in self.config[dataset]

        self.dataset = dataset
        self.pipetype = pipetype
        self.pipefunction = self.config[dataset][pipetype]

    def CRASH_load(self) -> pd.DataFrame:
        return load_crash_data()

    def CRASH_response_pipe(self, data=None, as_series=False):
        if data is None:
            data = self.CRASH_load()
        data = consolidate_response(data)
        if as_series:
            return data['Y']
        return data

    def CRASH_feature_pipe(self, data=None, select_feat=None):
        if data is None:
            data = self.CRASH_response_pipe()

        X_all = self.feature_builder.build_features(data)

        if select_feat is None:
            print(f'Using default feature selection for {self.dataset}, mode: {self.config[self.dataset]["features"]}')
            select_feat = self.config[self.dataset]["features"]
        try:
            X_selected = select_features(X_all, self.dataset, select_feat)
            return X_selected
        except ValueError as e:
            print(f"feature selector errored on {self.dataset} with features {select_feat}")
            raise
        
    def CRASH_full_pipe(self, as_df=False, select_feat=None):
        df_with_Y = self.CRASH_response_pipe()
        X = self.CRASH_feature_pipe(data=df_with_Y, select_feat=select_feat)
        Y = df_with_Y["Y"].values
        if as_df:
            return (X, Y)
        else: 
            return (X.to_numpy(), Y)

    def pipefunc(self):
        return self.pipefunction

    def __call__(self, *args, **kwargs):
        print(f"Using function {self.pipefunction.__name__}")
        return self.pipefunction(*args, **kwargs)
