import os
from typing import Any
from joblib import dump, load


def save_pipeline_joblib(pipeline: Any,
                         model_path: str):
    dump(pipeline, model_path)


def load_pipeline_joblib(model_path: str):
    pipeline = load(model_path)
    return pipeline
