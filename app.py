import copy
from urllib import response

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
import json
import jwt
import os
from dotenv import load_dotenv
import  psycopg2
import bcrypt
load_dotenv()
app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:3000"])
pipeline = joblib.load('./TrainedModels/xgboostmodel_tuned.pkl')
secret=os.getenv("SECRET_KEY")
DATABASE_URL=os.getenv("DATABASE_URL")
print("SECRET KEY:", secret)
@app.route('/')
def hello_world():
    return 'Hello World!'

models = {
    "linear_regression": joblib.load("TrainedModels/linearmodel.pkl"),
    "ridge": joblib.load("TrainedModels/ridgemodel.pkl"),
    "lasso": joblib.load("TrainedModels/lasssomodel.pkl"),
    "random_forest": joblib.load("TrainedModels/randommodel.pkl"),
    "xgboost": joblib.load("TrainedModels/xgboostmodel_tuned.pkl"),
}


with open("predictions.json","r")as f:
    metadeta = json.load(f)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])

    response=copy.deepcopy(metadeta)
    for name, pipeline in models.items():
        log_price = pipeline.predict(input_df)[0]
        price = np.expm1(log_price)
        response["models"][name]["prediction"] = round(float(price), 2)

    return jsonify(response)



@app.before_request
def checkauth():
    print("Testing if this middleware is  running or not")
    print("endpoint",request.endpoint)
    print("cookies",request.cookies)
    public_endpoints={
        "login",
        "register"
    }

    if request.method == "OPTIONS":
        print("OPTIONS REQUEST - SKIPPING AUTH")
        return None
    if request.endpoint in public_endpoints:
        return

    token=request.cookies.get("access_token")

    if not token:
        return jsonify({
            "message":"Authentication Required",
        }),401

    try:
        decoded=jwt.decode(
            token,
            secret,
            algorithms=["HS256"]
        )

        request.user_id=decoded["Userid"]

    except jwt.InvalidTokenError:
        return jsonify({
            "message":"Invalid Token Error",
        }),401







@app.route("/chart-data")
def chart_data():
    df = pd.read_csv("car.csv")
    data = df[["km_driven", "selling_price"]].sample(500,random_state=42).to_dict(orient="records")

    return jsonify(data)



@app.route("/data-car_names")
def car_data():
    rawdf = pd.read_csv("car.csv")
    # print(rawdf["fuel_type"].unique())
    # print(rawdf["seller_type"].unique())
    # print(rawdf.shape)
    # print(len(rawdf))
    unique_cars = rawdf['car_name'].dropna().unique()
    # data = rawdf[['car_name']].drop_duplicates().to_dict(orient="records")
    data = [{"value": car, "label": car} for car in unique_cars]

    return jsonify(data)

@app.route("/data-brand_names")
def brand_data():
    rawdf = pd.read_csv("car.csv")
    unique_brands = rawdf['brand'].dropna().unique()
    data = [{"value": brand, "label": brand} for brand in unique_brands]

    return jsonify(data)


@app.route("/data-model_names")
def model_data():
    rawdf = pd.read_csv("car.csv")
    unique_models = rawdf['model'].dropna().unique()
    data = [{"value": model, "label": model} for model in unique_models]

    return jsonify(data)


@app.route("/data-feature-importance")
def feature_importance():
    with open("feature_importance.json", "r") as f:
        data = json.load(f)

    return jsonify(data)



@app.route("/data-pricecomparison")
def price_comparison():
    with open("price_comparison_data.json", "r") as f:
        data=json.load(f)
    return jsonify(data)


@app.route("/data-tabledata")
def table_data():
    with open("model_comparison.json", "r") as f:
        data=json.load(f)
    return jsonify(data)


@app.route("/login",methods=['POST'])
def login():
    data=request.get_json()
    Username=data["Username"]
    Email=data["Email"]
    Password=data["Password"]
    print("Username:", repr(Username))
    print("Email:", repr(Email))
    print("Password:", repr(Password))

    # checking if username and password entered by user is correct
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cursor:
            cursor.execute("Select  *from  users where Email=%s ",
                           ( Email,))
            userfoundstatus=cursor.fetchone()

    if not userfoundstatus:
        print("User not found")
    userid=userfoundstatus[0]
    dbusername=userfoundstatus[1]
    dbemail=userfoundstatus[2]
    dbpassword=userfoundstatus[3]

    if not bcrypt.checkpw(Password.encode("utf-8"), dbpassword.encode("utf-8")):
        return jsonify({
            "message":"Incorrect Password",
        })


    if  Email!=dbemail:
        return jsonify({
            "message":"invalid credentials",
        })
                


    token=jwt.encode({"Userid":1},secret,algorithm="HS256")
    response=jsonify({
       "message":"login successful",
     })
    response.set_cookie(
        "access_token",
         token,
        httponly=True,
         samesite="lax"
     )
    return response,200

@app.route("/register",methods=['POST'])
def register():
    data=request.get_json()
    Username=data.get("Username","").strip()
    Email=data.get("Email","").strip()
    Password=data.get("Password","").strip()
    print(data)
    if not Username :
        return jsonify({"message":"username is required"}),400

    if not Email:
        return jsonify({"message":"email is required"}),400

    if not "@" in Email or not "." in Email:
        return jsonify({"message":"invalid email"}),400

    if not Password:
        return jsonify({"message":"password is required"}),400

    password_hash=bcrypt.hashpw(
        Password.encode("utf-8"),
        bcrypt.gensalt()
    )
    password_hash_str=password_hash.decode("utf-8")
    print("Type being inserted:", type(password_hash_str))
    with psycopg2.connect(DATABASE_URL) as conn:

        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO users (username,email,password_hash) values (%s,%s,%s)",(Username,Email,password_hash_str))

if __name__ == '__main__':
    app.run()
