from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()
model_path = "titanic_model.pkl"

app = FastAPI()
app = FastAPI()

class Passenger(BaseModel):
    Pclass: int  # 1st, 2nd, 3rd class
    Sex: str     # 'male' or 'female'
    Age: float
    Fare: float


def train_and_save_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]].dropna()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    X = df[["Pclass", "Sex", "Age", "Fare"]]
    y = df["Survived"]
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model


# Load or train model
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = train_and_save_model()


@app.get("/")
def read_root():
    return {"message": "Titanic Survival Predictor API"}


@app.post("/predict/")
def predict_survival(passenger: Passenger):
    sex = 0 if passenger.Sex.lower() == "male" else 1
    data = [[passenger.Pclass, sex, passenger.Age, passenger.Fare]]
    prediction = model.predict(data)[0]
    result = "Survived" if prediction == 1 else "Did not survive"
    return {"prediction": int(prediction), "result": result}


@app.post("/retrain/")
def retrain():
    global model
    model = train_and_save_model()
    return {"message": "Model retrained successfully"}
@app.post("/tesst/")
def retrain():
    global model
    model = train_and_save_model()
    return {"message": "Model retrained successfully"}