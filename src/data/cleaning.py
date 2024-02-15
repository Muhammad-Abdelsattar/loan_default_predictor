from typing import Optional
import pandas as pd


def get_highly_inbalanced_categories(data,threshold: float):
    cat_desc = data.describe(include=[object])
    cat_dict = dict(cat_desc.loc["freq"] > threshold * cat_desc.loc["count"])
    highly_inbalanced = []
    for key in cat_dict.keys():
        if(cat_dict[key]):
            highly_inbalanced.append(key)
    return highly_inbalanced

def drop_ininformatve_columns(data: pd.DataFrame):
    data.drop(["ID","year"], axis=1, inplace=True)
    return data

def impute_column_by_group(data,column,group):
    groups_dict = data.groupby(group)[column].median().to_dict() 
    data[column] = data[column].fillna(data[group].map(groups_dict))
    return data, {column:{group:groups_dict}} #useful for imputing test data

def impute_training_numerical_features(data,groups_tuple):
    imputation_list = []
    for c,g in groups_tuple:
        data, d = impute_column_by_group(data,c,g)
        imputation_list.append(d)
    return data,imputation_list

def impute_test_numerical_features(test_data: pd.DataFrame, imputation_dict: dict):
    for element in imputation_dict:
        column = list(element.keys())[0]
        group = list(element[column].keys())[0]
        test_data[column] = test_data[column].fillna(test_data[group].map(element[column][group]))
    return test_data

def impute_training_categorical_features(data: pd.DataFrame):
    cat_imputation_dict = {}
    cat_cols = list(data.select_dtypes(include=[object]).columns)
    for col in cat_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
        cat_imputation_dict[col] = data[col].mode()[0]
    return data,cat_imputation_dict

def impute_test_categorical_features(test_data: pd.DataFrame, cat_imputation_dict: dict):
    cat_cols = list(test_data.select_dtypes(include=[object]).columns)
    for col in cat_cols:
        test_data[col].fillna(cat_imputation_dict[col], inplace=True)
    return test_data


def handle_training_missing_values(data: pd.DataFrame, numerical_imputation_metadata: list):
    data, num_imputation_dict = impute_training_numerical_features(data=data, groups_tuple=numerical_imputation_metadata)
    data, cat_imputation_dict = impute_training_categorical_features(data=data)
    return data, num_imputation_dict, cat_imputation_dict

def handle_test_missing_values(test_data: pd.DataFrame, num_imputation_dict: dict, cat_imputation_dict: dict):
    test_data = impute_test_categorical_features(test_data, cat_imputation_dict)
    test_data = impute_test_numerical_features(test_data, num_imputation_dict)
    return test_data
    

def clean_training_data(data: pd.DataFrame, threshold: float, numerical_imputation_metadata):
    data = drop_ininformatve_columns(data)
    cols_to_drop = get_highly_inbalanced_categories(data, threshold)
    data.drop(cols_to_drop,axis=1)
    data, num_imputation_dict, cat_imputation_dict = handle_training_missing_values(data, numerical_imputation_metadata)
    return data, cols_to_drop, num_imputation_dict, cat_imputation_dict

def clean_test_data(test_data: pd.DataFrame, columns_to_drop: list, num_imputation_dict, cat_imputation_dict):
    test_data = drop_ininformatve_columns(test_data)
    test_data.drop(columns_to_drop,axis=1)
    test_data = handle_test_missing_values(test_data, num_imputation_dict, cat_imputation_dict)
    return test_data
