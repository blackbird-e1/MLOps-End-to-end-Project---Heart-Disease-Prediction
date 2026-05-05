# from pydantic import BaseModel

# class InputData(BaseModel):
#     data: list


# @app.post("/predict")
# def predict(input: InputData):
#     arr = np.array(input.data).reshape(1, -1)
#     prediction = model.predict(arr)

#     return {
#         "prediction": int(prediction[0]),
#         "result": "Disease" if prediction[0] == 1 else "Healthy"
#     }