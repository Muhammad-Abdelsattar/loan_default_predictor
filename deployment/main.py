from fastapi import FastAPI
from pydantic import BaseModel
from model import *


app = FastAPI()
model = load_model()

@app.get("/")
def index():
    return "Loan default prediction API."


@app.post("/predict")
def predict(inp: dict):
    features = prepare_input([inp["inp"]])
    return {"prediction": model.predict(features).tolist()}

