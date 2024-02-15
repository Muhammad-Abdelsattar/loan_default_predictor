import os
from omegaconf import OmegaConf
# import argparse
import pandas as pd 
from data.utils import split_features_label
from modeling.utils import load_pipeline_joblib
from evaluation.evaluation import evaluate_pipeline
from evaluation.utils import write_scores

# parser = argparse.ArgumentParser()
# parser.add_argument("config",type=str, help="Params.yaml file path.")

def load_data():
    clean_test_data_path = config["data"]["path"]+"/clean_test.csv"
    test_data = pd.read_csv(clean_test_data_path)
    test_dict = split_features_label(test_data,"Status")
    return test_dict

def evaluate(config: dict):
    test_dict = load_data()
    pipeline = load_pipeline_joblib(config["pipeline"]["path"])
    scores = evaluate_pipeline(pipeline, test_dict)
    print(scores)
    scores_path = config["eval"]["metrics"]
    write_scores(scores, scores_path)
    return 


if __name__ =="__main__":
    # args = parser.parse_args()
    config = OmegaConf.load("params.yaml")
    evaluate(config)

