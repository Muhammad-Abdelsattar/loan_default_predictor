from joblib import load
import pandas as pd 

def load_model():
    model = load("artifacts/models/model.joblib")
    return model

def prepare_input(inp: list[dict]):
    return pd.DataFrame(inp)