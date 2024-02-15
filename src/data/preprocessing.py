from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (PowerTransformer,
                                   OneHotEncoder,
                                   MinMaxScaler,
                                   StandardScaler)


numerical_transformers = {"MinMax": MinMaxScaler(),
                          "PowerTransformer": PowerTransformer(),
                          "StandardScaler": StandardScaler()}

categorical_transformers = {"OneHotEncoder": OneHotEncoder()}

all_transformers = {"numerical": numerical_transformers,
                    "categorical": categorical_transformers}


def build_transformer(transformer_type: str, transformer_name: str, columns: list):
    transformer = all_transformers[transformer_type][transformer_name]
    return (transformer_name, transformer, columns)


def build_preprocessor(transforms_config: dict):
    transformers = []
    for transform_type in transforms_config.keys():
        for t in transforms_config[transform_type].keys():
            transformers.append(build_transformer(transform_type, t, list(transforms_config[transform_type][t])))
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor
