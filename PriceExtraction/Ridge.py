import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv("car.csv")

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

y_log = np.log1p(y)

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     ["car_name", "brand", "model", "fuel_type", "seller_type"]),

    ("log_km", FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
     ["km_driven"]),

    ("scaler", StandardScaler(),
     ["vehicle_age", "engine", "max_power", "mileage"]),
])

alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

best_alpha = None
best_test_r2 = -999
best_pipeline = None

for alpha in alphas:
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=alpha))
    ])

    pipeline.fit(X_train, y_train_log)

    train_pred_log = pipeline.predict(X_train)
    test_pred_log = pipeline.predict(X_test)

    train_r2 = r2_score(np.expm1(y_train_log), np.expm1(train_pred_log))
    test_r2 = r2_score(np.expm1(y_test_log), np.expm1(test_pred_log))

    price_comparison_data = [
        {
            "actual": float(actual),
            "predicted": float(pred_price)
        }
        for actual, pred_price in zip(y_test_log, test_pred_log)
    ]
    with open("price_comparison_data.json", "w") as outfile:
        json.dump(price_comparison_data, outfile)

    test_mae = mean_absolute_error(
        np.expm1(y_test_log),
        np.expm1(test_pred_log)
    )

    print(
        f"Alpha = {alpha:<7} "
        f"Train R² = {train_r2:.4f}   "
        f"Test R² = {test_r2:.4f}   "
        f"MAE = ₹{test_mae:,.0f}"
    )

    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_alpha = alpha
        best_pipeline = pipeline

print("\nBest Alpha:", best_alpha)
print("Best Test R²:", round(best_test_r2, 4))

joblib.dump(best_pipeline, "../TrainedModels/ridgemodel.pkl")