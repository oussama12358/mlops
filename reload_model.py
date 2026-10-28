
import mlflow
import mlflow.sklearn
import pandas as pd

# 1. Find the best run
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)
run_id = runs.iloc[0]["run_id"]

# 2. Reload the model saved in that run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
print("Model reloaded :", type(model).__name__)
print("Number of expected variables :", model.n_features_in_)

# 3. Create a new customer
customer = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,
}])

# 4. Keep only the features used by this model
customer = customer[list(model.feature_names_in_)]

# 5. Predict: 1 = leaves, 0 = stays
print(
    "Will the customer leave? ->",
    "YES" if model.predict(customer)[0] == 1 else "NO"
)
