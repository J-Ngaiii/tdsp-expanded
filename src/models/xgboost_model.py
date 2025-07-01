import mlflow
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from itertools import product
import os

from src.features.preprocessing import Preprocessing


def evaluate_fold(model, X_val, y_val):
    y_pred = model.predict(xgb.DMatrix(X_val))
    y_pred_labels = (y_pred >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_val, y_pred),
        "accuracy": accuracy_score(y_val, y_pred_labels),
        "f1": f1_score(y_val, y_pred_labels),
        "precision": precision_score(y_val, y_pred_labels),
        "recall": recall_score(y_val, y_pred_labels),
        "conf_matrix": confusion_matrix(y_val, y_pred_labels)
    }


def cross_validate(params, X, y, k=5, return_final_model=False):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    all_metrics = []
    all_conf_matrices = []
    last_model = None

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, dtrain, num_boost_round=100)

        fold_metrics = evaluate_fold(model, X_val, y_val)
        all_conf_matrices.append(fold_metrics.pop("conf_matrix"))
        all_metrics.append(fold_metrics)

        last_model = model

    avg_metrics = {
        metric: np.mean([fold[metric] for fold in all_metrics])
        for metric in all_metrics[0].keys()
    }
    avg_conf_matrix = np.mean(all_conf_matrices, axis=0).round().astype(int)

    if return_final_model:
        return avg_metrics, avg_conf_matrix, last_model
    else:
        return avg_metrics, avg_conf_matrix


def log_confusion_matrix(conf_matrix, run_id):
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Avg Confusion Matrix (5-fold)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs("plots", exist_ok=True)
    path = f"plots/conf_matrix_{run_id}.png"
    plt.savefig(path)
    mlflow.log_artifact(path, artifact_path="confusion_matrices")
    plt.close()


def train_xgboost(dataset, verbose=True):
    # Load dataset and preprocess
    if verbose:
        print("Beginning preprocessing")
    if isinstance(dataset, str):
        try:
            processor = Preprocessing(dataset, 'full_pipe')
            X, y = processor()
        except ValueError as e:
            print(f"Errored on processing with dataset {dataset}, passing error:\n{e}")
            raise
    else:
        raise TypeError("Currently, only string dataset identifiers are supported.")
    
    param_grid = {
        "max_depth": [3, 5],
        "eta": [0.1, 0.3],
        "subsample": [0.8, 1.0],
        "objective": ["binary:logistic"],
        "eval_metric": ["logloss"],
        "verbosity": [0],
    }
    if verbose:
        print(f"Preprocessing complete! Training on param grad with 5-fold CV:\n{param_grid}")

    best_score = -float("inf")
    best_model = None
    best_params = None
    best_metrics = None

    keys, values = zip(*param_grid.items())
    for v in product(*values):
        params = dict(zip(keys, v))

        with mlflow.start_run() as run:
            mlflow.log_params(params)

            metrics, avg_conf_matrix, fitted_model = cross_validate(params, X, y, k=5, return_final_model=True)

            mlflow.log_metrics(metrics)
            log_confusion_matrix(avg_conf_matrix, run.info.run_id)

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model = fitted_model
                best_params = params
                best_metrics = metrics

            if verbose:
                print(f"Completed Run\nParams: {params} => Metrics: {metrics}")

    if verbose:
        print("\n✅ Best Model:")
        print(f"Params: {best_params}")
        print(f"Metrics: {best_metrics}")

    return best_model
