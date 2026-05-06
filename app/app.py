# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
# import joblib
# import numpy as np

# app = FastAPI()

# # Load model
# model = joblib.load("models/model.pkl")

# class InputData(BaseModel):
#     data: List[float]


# @app.get("/")
# def home():
#     return {"message": "Heart Disease Prediction API is running"}


# @app.post("/predict")
# def predict(input: InputData):
#     arr = np.array(input.data).reshape(1, -1)
#     prediction = model.predict(arr)

#     return {
#         "prediction": int(prediction[0]),
#         "result": "Disease" if prediction[0] == 1 else "Healthy"
#     }

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

# NEW
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Load model
model = joblib.load("models/model.pkl")


class InputData(BaseModel):
    data: List[float]


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(input: InputData):

    arr = np.array(input.data).reshape(1, -1)

    prediction = model.predict(arr)

    return {
        "prediction": int(prediction[0]),
        "result": "Disease" if prediction[0] == 1 else "Healthy"
    }


# NEW
Instrumentator().instrument(app).expose(app)