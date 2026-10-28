# ===== 1. Import the libraries =====
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn API")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/churn-model@champion")


class Customer(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


@app.get("/")
def root():
    return {
        "message": "Churn API is running",
        "docs": "/docs",
        "health": "/health",
        "prediction": "POST /predict",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])
    row = pd.get_dummies(row)
    row = row.reindex(columns=model.feature_names_in_, fill_value=0)
    result = int(model.predict(row)[0])
    return {
        "churn": result,
        "message": "The customer will leave" if result == 1 else "The customer stays",
    }