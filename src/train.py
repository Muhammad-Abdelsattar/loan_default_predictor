from omegaconf import OmegaConf
import argparse
import pandas as pd 
from data.utils import split_features_label
from training.training import train_pipeline
from data.preprocessing import build_preprocessor
from modeling.model_factory import build_model, build_pipeline
from modeling.utils import save_pipeline_joblib

# parser = argparse.ArgumentParser()
# parser.add_argument("config",type=str, help="Params.yaml file path.")


def load_data():
    clean_training_data_path = config["data"]["path"]+"/clean_train.csv"
    clean_test_data_path = config["data"]["path"]+"/clean_test.csv"
    train_data = pd.read_csv(clean_training_data_path)
    test_data = pd.read_csv(clean_test_data_path)
    train_dict = split_features_label(train_data,"Status")
    test_dict = split_features_label(test_data,"Status")
    return train_dict, test_dict

def train(config: dict):
    preprocessor = build_preprocessor(config["pipeline"]["preprocessor"])
    model = build_model(config["pipeline"]["model"])
    pipeline = build_pipeline(preprocessor, model)
    training_data, test_data = load_data()
    pipeline, score = train_pipeline(training_data, pipeline, config["train"]["num_folds"], config["train"]["scoring"])
    save_pipeline_joblib(pipeline, config["pipeline"]["path"])
    print(config["train"]["scoring"]+" : "+str(score))


if __name__ == "__main__":
    # args = parser.parse_args()
    config = OmegaConf.load("params.yaml")
    train(config)
