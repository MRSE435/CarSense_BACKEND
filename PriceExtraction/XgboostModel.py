import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import json

# ---------- Load data ----------
df = pd.read_csv("car.csv")
X = df.drop("selling_price", axis=1)
y = df["selling_price"]
y_log = np.log1p(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# ---------- Preprocessing ----------
tf1 = ColumnTransformer([
    ("Onehotencoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     ["car_name", "brand", "model", "fuel_type", "seller_type","transmission_type"]),
    ("Logtransformer", FunctionTransformer(np.log1p,feature_names_out="one-to-one" ), ["km_driven"]),
    ("StandardScaler", StandardScaler(), ["vehicle_age", "engine", "max_power", "mileage"]),
])

# ---------- Pipeline (no early stopping here — search handles n_estimators directly) ----------
pipeline = Pipeline([
    ("preprocessor", tf1),
    ("model", XGBRegressor(random_state=42, eval_metric="rmse")),
])

# ---------- Parameter search space ----------
param_dist = {
    "model__n_estimators": [200, 400, 600, 800, 1000],
    "model__max_depth": [3, 4, 5, 6, 7],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8],
    "model__colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8],
    "model__min_child_weight": [1, 3, 5, 7, 10],
    "model__reg_alpha": [0, 0.01, 0.1, 0.5, 1, 2],
    "model__reg_lambda": [0.5, 1, 5, 10, 20],
    "model__gamma": [0, 0.1, 0.3, 0.5, 1],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,                # number of random combos to try — raise for a more thorough search
    scoring="r2",
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=2,
)

search.fit(X_train, y_train_log)

print("\nBest params:", search.best_params_)
print("Best CV R2:", search.best_score_)

# ---------- Evaluate best model on held-out test set ----------
best_pipeline = search.best_estimator_

train_pred = np.expm1(best_pipeline.predict(X_train))
test_pred = np.expm1(best_pipeline.predict(X_test))
y_train_price = np.expm1(y_train_log)
y_test_price = np.expm1(y_test_log)



price_comparison_data=[
    {
        "actual": float(actual),
        "predicted": float(pred_price)
    }
    for actual, pred_price in zip(y_test_price, test_pred)
]
with open("price_comparison_data.json", "w") as outfile:
    json.dump(price_comparison_data, outfile)

print("\n--- Final tuned XGBoost ---")
print("R2 train:", round(r2_score(y_train_price, train_pred), 4))
print("R2 test:", round(r2_score(y_test_price, test_pred), 4))
print("RMSE-test",round(root_mean_squared_error(y_test_price, test_pred), 4))
print("MAE:", round(mean_absolute_error(y_test_price, test_pred), 2))


import joblib
joblib.dump(best_pipeline, "../TrainedModels/xgboostmodel_tuned.pkl")
print("\nSaved best pipeline to xgboostmodel_tuned.pkl")
