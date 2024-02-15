from .utils import *

def evaluate_pipeline(pipeline, test_data):
    y_pred = pipeline.predict(test_data["features"])
    scores = compute_eval_scores(test_data["label"], y_pred)
    return scores