from fastapi import FastAPI
import uvicorn
import pickle
from pydantic import BaseModel


class Diabetes(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


app = FastAPI()

diabetes_model = open("dbts_model.pkl", "rb")
model = pickle.load(diabetes_model)


@app.get("/")
def read_root():
    return {"welcome to the homepage of the api"}


@app.post("/predict")
def get_diabetes_result(data: Diabetes):
    received = data.dict()
    Pregnancies = received["Pregnancies"]
    Glucose = received["Glucose"]
    BloodPressure = received["BloodPressure"]
    SkinThickness = received["SkinThickness"]
    Insulin = received["Insulin"]
    BMI = received["BMI"]
    DiabetesPedigreeFunction = received["DiabetesPedigreeFunction"]
    Age = received["Age"]

    pred = model.predict([[Pregnancies,
                           Glucose,
                           BloodPressure,
                           SkinThickness,
                           Insulin,
                           BMI,
                           DiabetesPedigreeFunction,
                           Age]]).tolist()
    if (pred[0] == 1):
        pred = "Probably have diabetes😥"
    elif(pred[0] == 0):
        pred = "No diabetes🍬🍦🍩"

    return {"Prediction": pred}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=3000, debug=True)
