import pandas as pd
import numpy as np
from src.features.preprocessing import Preprocessing
from src.models.linear_model import train_logistic

if __name__ == "__main__":
    processor = Preprocessing('nyc-crashes', 'full_pipe')
    X, Y = processor(as_df=False)
    best_lr = train_logistic(dataset='nyc-crashes')
