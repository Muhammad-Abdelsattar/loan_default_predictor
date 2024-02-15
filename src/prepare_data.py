import os

print(os.getcwd())
import pandas as pd 
from omegaconf import OmegaConf
import argparse
from data.cleaning import clean_training_data, clean_test_data
from data.utils import write_csv_data


# parser = argparse.ArgumentParser()
# parser.add_argument("config",type=str, help="Params.yaml file path.")

def prepare_data(config: dict):
    train_data = pd.read_csv(config["data"]["path"]+"/train.csv")
    test_data = pd.read_csv(config["data"]["path"]+"/test.csv")
    clean_train, cols_to_drop, num_imp, cat_imp = clean_training_data(train_data,0.95,config["data"]["impute"]["numerical"])
    clean_test = clean_test_data(test_data,cols_to_drop,num_imp,cat_imp)
    write_csv_data(clean_train, config["data"]["path"]+"/clean_train.csv")
    write_csv_data(clean_test, config["data"]["path"]+"/clean_test.csv")
    return 

if __name__ == "__main__":
    # args = parser.parse_args()
    config = OmegaConf.load("params.yaml")
    prepare_data(config)
    