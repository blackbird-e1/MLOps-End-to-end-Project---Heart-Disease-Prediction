# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import os

# Load data
df = pd.read_csv("data/heart_disease_data.csv")

X = df.drop(columns='target')
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# Hyperparameter tuning
# param_grid = {
#     'C': [0.1, 1, 10],
#     'solver': ['liblinear', 'lbfgs']
# }
param_grid = {
    'model__C': [0.1, 1, 10],
    'model__solver': ['liblinear', 'lbfgs']
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000,  random_state=42))
])

# grid = GridSearchCV(LogisticRegression(), param_grid, cv=3)
# grid.fit(X_train, y_train)

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Heart_Disease_Project")

with mlflow.start_run():
    mlflow.log_params(grid.best_params_)
    # mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("cv_score", grid.best_score_)
    mlflow.sklearn.log_model(best_model, "model")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")
# joblib.dump(best_model, "models/model.pkl")