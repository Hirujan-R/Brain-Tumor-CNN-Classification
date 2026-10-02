"""MLflow configuration helpers."""

import os
from pathlib import Path


DEFAULT_EXPERIMENT = "Brain_Tumor_CV"


def setup_mlflow(tracking_uri: str | None = None, experiment_name: str = DEFAULT_EXPERIMENT):
    """
    Configure the MLflow tracking URI and experiment.

    Resolution order for the tracking URI:
        1. Explicit ``tracking_uri`` argument.
        2. ``MLFLOW_TRACKING_URI`` environment variable.
        3. Local ``./mlruns`` file store (safe default for Colab / offline use).
    """
    import mlflow

    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if tracking_uri is None:
        local_store = Path("mlruns").resolve().as_uri()
        tracking_uri = local_store

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow
