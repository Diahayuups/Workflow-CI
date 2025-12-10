import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# =============================
# 1. PARSE INPUT ARGUMENT
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
# 4. EVALUATION (PRINT ONLY)
# =============================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print("Best Parameters:", grid.best_params_)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("ROC AUC:", roc)

# =============================
# Save Model
# =============================
import os

# folder khusus untuk output
OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_path = f"{OUTPUT_DIR}/best_random_forest.pkl"
joblib.dump(best_model, model_path)

mlflow.log_artifact(model_path)

print(f"Model saved successfully at {model_path}")
