import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor
import pandas as pd
import os

app = FastAPI()

current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "sleepquality.cbm")

model = CatBoostRegressor()
model.load_model(model_path)

class InputData(BaseModel):
    gender: str
    age: int
    occupation: str
    sleep_duration: float
    physical_activity_level: int
    bmi_category: str
    heart_rate: int
    daily_steps: int
    sleep_disorder: str

def predictSleep(data: InputData):
    input_df = {
        'Gender': data.gender,
        'Age': data.age,
        'Occupation': data.occupation,
        'Sleep Duration': data.sleep_duration,
        'Physical Activity Level': data.physical_activity_level,
        'BMI Category': data.bmi_category,
        'Heart Rate': data.heart_rate,
        'Daily Steps': data.daily_steps,
        'Sleep Disorder': data.sleep_disorder
    }
    input_df = pd.DataFrame([input_df])
    print(input_df)
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    input_df[categorical_cols] = input_df[categorical_cols].astype('category')
    print(input_df)
    prediction = model.predict(input_df)
    print(prediction)
    # prediction = "hello"
    return {"prediction": prediction[0]}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(data: InputData):
    return predictSleep(data)
