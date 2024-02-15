import os
import json
from sklearn.metrics import (f1_score,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             roc_auc_score)


def compute_eval_scores(y_true, y_pred):
    scores = {}
    scores["accuracy"] = float(accuracy_score(y_pred, y_true))
    scores["recall"] = float(recall_score(y_true, y_pred))
    scores["precision"] = float(precision_score(y_true, y_pred))
    scores["f1_score"] = float(f1_score(y_true, y_pred))
    scores["AUC"] = float(roc_auc_score(y_true, y_pred))
    return scores

def write_scores(scores: dict, file_path: str):
    with open(file_path, "w") as f:
        json.dump(scores, f)