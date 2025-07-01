import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class CrashFeatureBuilder:
    def __init__(self):
        self.encoder = OneHotEncoder(drop='first', sparse_output=False)

    def build_features(self, data):
        copy = data.copy()
        
        # Step 1: Select columns
        cols = ['crash_time', 'borough', 'vehicle_type_code1', 'contributing_factor_vehicle_1']
        copy = copy[cols].copy()

        # Step 2: Parse time
        crash_time_parsed = pd.to_datetime(copy['crash_time'], format='%H:%M', errors='coerce')
        hour_of_day = crash_time_parsed.dt.hour.fillna(0).astype(int)

        # Step 3: Normalize categories
        copy['borough'] = copy['borough'].fillna('Unspecified')
        top_vehicles = copy['vehicle_type_code1'].value_counts().nlargest(10).index
        top_factors = copy['contributing_factor_vehicle_1'].value_counts().nlargest(10).index

        copy['vehicle_type_code1'] = copy['vehicle_type_code1'].apply(lambda x: x if x in top_vehicles else 'Other')
        copy['contributing_factor_vehicle_1'] = copy['contributing_factor_vehicle_1'].apply(
            lambda x: x if x in top_factors else 'Other'
        )

        # Step 4: Peak hour diff
        peak_hour = hour_of_day.value_counts().idxmax()
        copy['time_from_peak_hour'] = hour_of_day - peak_hour

        # Step 5: One-hot encode
        cat_cols = ['borough', 'vehicle_type_code1', 'contributing_factor_vehicle_1']
        encoded = self.encoder.fit_transform(copy[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(cat_cols), index=copy.index)

        # Step 6: Combine
        X = pd.concat([copy[['time_from_peak_hour']], encoded_df], axis=1)
        return X
