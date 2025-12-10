import argparse
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# =============================
# 1. PARSE ARGUMENT
# =============================
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="dataset/telco_processed.csv")
args = parser.parse_args()

print("DATA PATH:", args.data_path)

# =============================
# 2. LOAD DATASET
# =============================
df = pd.read_csv(args.data_path)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 3. GRIDSEARCH TUNING
# =============================
model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# =============================
# 4. MLFLOW LOGGING (LOCAL ONLY)
# =============================
with mlflow.start_run():

    mlflow.log_params(grid.best_params_)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)

    mlflow.log_metric("true_negative", cm[0][0])
    mlflow.log_metric("false_positive", cm[0][1])
    mlflow.log_metric("false_negative", cm[1][0])
    mlflow.log_metric("true_positive", cm[1][1])

    # save model
    joblib.dump(best_model, "best_random_forest.pkl")
    mlflow.log_artifact("best_random_forest.pkl")

print("TRAINING SELESAI.")
