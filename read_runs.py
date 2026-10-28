import mlflow

# Read the runs of our experiment, sorted by F1 (best first)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)
  
# Display the most useful columns
columns = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

print(runs[columns].to_string(index=False))

# Display the best run
best = runs.iloc[0]
print()
print("Best run :", best["tags.mlflow.runName"],
      "| id :", best["run_id"][:8], "...")
