"""Backwards-compatible re-export of the MLflow setup helper."""

from src.mlops.mlflow_setup import DEFAULT_EXPERIMENT, setup_mlflow

__all__ = ["DEFAULT_EXPERIMENT", "setup_mlflow"]
