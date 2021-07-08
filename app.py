from fastapi import FastAPI
import uvicorn
import pickle
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


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

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

diabetes_model = open("dbts_model.pkl", "rb")
model = pickle.load(diabetes_model)


@app.get("/")
def read_root():
    return {"message": "welcome to the homepage of the api"}


@app.post("/predict")
def get_diabetes_result(data: Diabetes):
    pred_data = [[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]]

    pred = model.predict(pred_data)
    if (pred[0] == 1):
        pred = "Probably have diabetes😥"
    elif(pred[0] == 0):
        pred = "No diabetes🍬🍦🍩"

    return {"Prediction": pred}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
