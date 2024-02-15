from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


model_dispatcher = {"RF": RandomForestClassifier,
                    "LR": LogisticRegression,
                    "XGB": XGBClassifier,
                    "LGBM": LGBMClassifier}


def build_model(config: dict):
    return model_dispatcher[config["name"]](**config["params"],random_state=42)


def build_pipeline(preprocessor, model):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    return pipeline 