"""Export all runs from an MLflow experiment to a CSV file."""

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MLflow runs to CSV.")
    parser.add_argument("--experiment", type=str, default="Brain_Tumor_CV")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="Defaults to $MLFLOW_TRACKING_URI or local ./mlruns.",
    )
    parser.add_argument("--output", type=str, default="brain_tumor_cv_runs.csv")
    return parser.parse_args()


def main():
    import mlflow

    args = parse_args()

    if args.tracking_uri is None:
        args.tracking_uri = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(args.tracking_uri)

    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(
            f"Experiment '{args.experiment}' not found at {args.tracking_uri}. "
            "Train a model first or pass --experiment."
        )

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    runs.to_csv(args.output, index=False)
    print(f"Wrote {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
