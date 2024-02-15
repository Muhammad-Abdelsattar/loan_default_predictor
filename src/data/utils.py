import pandas as pd 
from sklearn.model_selection import train_test_split

def split_features_label(data: pd.DataFrame, label_name: str):
    data_dict = {}
    feature_columns = list(data.columns)
    feature_columns.remove(label_name)
    data_dict["features"] = data[feature_columns]
    data_dict["label"] = data[label_name].values
    return data_dict

# def split_data(features, label, percentage: float = 0.2, stratify: str = None):
#     x_train,x_test,y_train,y_test = train_test_split(features,label,stratify=stratify,test_size=percentage)
#     return {"x_train":x_train, "x_test":x_test, "y_train":y_train, "y_test":y_test}

def write_csv_data(data: pd.DataFrame, path: str):
    data.to_csv(path,index=False)
    return 