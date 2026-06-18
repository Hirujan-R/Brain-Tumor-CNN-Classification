def main():
  import mlflow

  experiment = mlflow.get_experiment_by_name("Brain_Tumor_CV")

  runs = mlflow.search_runs(
      experiment_ids=[experiment.experiment_id]
  )

  runs.to_csv("brain_tumor_cv_runs.csv", index=False)

if __name__ == "__main__":
  main()
