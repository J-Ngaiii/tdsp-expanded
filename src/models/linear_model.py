import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from itertools import product
import os

from src.features.preprocessing import Preprocessing


def evaluate_fold(model, X_val, y_val):
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_val, y_proba),
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "conf_matrix": confusion_matrix(y_val, y_pred)
    }


def cross_validate(params, X, y, k=5, return_final_model=False):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    all_metrics = []
    all_conf_matrices = []

    best_model = None

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        fold_metrics = evaluate_fold(model, X_val, y_val)
        all_conf_matrices.append(fold_metrics.pop("conf_matrix"))
        all_metrics.append(fold_metrics)

        if return_final_model:
            best_model = model  # last fold model

    avg_metrics = {
        metric: np.mean([fold[metric] for fold in all_metrics])
        for metric in all_metrics[0].keys()
    }
    avg_conf_matrix = np.mean(all_conf_matrices, axis=0).round().astype(int)

    if return_final_model:
        return avg_metrics, avg_conf_matrix, best_model
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


def train_logistic(dataset, verbose=True):
    if verbose:
        print("Beginning preprocessing")
    X, y = Preprocessing(dataset, 'full_pipe')(as_df=False)

    param_grid = {
        "C": [0.1, 1.0, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [500],
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
