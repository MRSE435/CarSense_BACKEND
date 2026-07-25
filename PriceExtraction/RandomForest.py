import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
linearmodeldf = pd.read_csv("car.csv")
print(linearmodeldf.columns.tolist())



X = linearmodeldf.drop("selling_price", axis=1)
y = linearmodeldf["selling_price"]
# log transform target
y_log = np.log1p(y)
X_train, X_test, y_train_log, y_test_log = train_test_split(
X, y_log, test_size=0.2, random_state=42
)
tf1=ColumnTransformer([
    ("Onehotencoder",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),["car_name","brand","model","fuel_type","seller_type"]),
    ("Logtransformer",FunctionTransformer(np.log1p,feature_names_out="one-to-one"),["km_driven"]),
    ("StandardScaler",StandardScaler(),["vehicle_age","engine","max_power","mileage"]),
])
pipeline=Pipeline(
    [
        ("preprocessor",tf1),
        (
         "model",RandomForestRegressor()
        )
    ]
)


pipeline.fit(X_train, y_train_log)
# predict log price

# testing model with custom input

# custom_input = pd.DataFrame({
#     "car_name": ["Honda City"],
#     "brand": ["Honda"],
#     "model": ["City"],
#     "vehicle_age": [5],
#     "km_driven": [45000],
#     "seller_type": ["Dealer"],
#     "fuel_type": ["Petrol"],
#     "engine": [1498],
#     "mileage": [17.8],
#     "max_power": [117.3]
# })

custom_input = pd.DataFrame({
    "car_name": ["Mercedes-Benz C-Class"],
    "brand": ["Mercedes-Benz"],
    "model": ["C-Class"],
    "vehicle_age": [4],
    "km_driven": [38000],
    "seller_type": ["Dealer"],
    "fuel_type": ["Diesel"],
    "engine": [1950],
    "mileage": [19.0],
    "max_power": [191.0]
})
# calculating train r square
train_pred=pipeline.predict(X_train)


pred_log = pipeline.predict(custom_input)
pred_price = np.expm1(pred_log)

print(f"Predicted Price: ₹{pred_price[0]:,.0f}")

pred_log = pipeline.predict(X_test)
# convert back to actual price
pred_price = np.expm1(pred_log)
y_test_price = np.expm1(y_test_log)


price_comparison_data=[
    {
        "actual": float(actual),
        "predicted": float(pred_price)
    }
    for actual, pred_price in zip(y_test_price, pred_price)
]
with open("price_comparison_data.json", "w") as outfile:
    json.dump(price_comparison_data, outfile)


print("R2  test Score:", round(r2_score(y_test_price, pred_price), 4))
print("r2 train score",r2_score(np.expm1(train_pred),np.expm1(y_train_log)))
print("MAE:", round(mean_absolute_error(y_test_price, pred_price), 2))

joblib.dump(pipeline, "../TrainedModels/randommodel.pkl")
