from sklearn.model_selection import cross_val_score

def train_pipeline(data, pipeline, num_folds, scoring):
    cv_scores = cross_val_score(estimator=pipeline,
                                X=data["features"],
                                y=data["label"],
                                cv=num_folds,
                                scoring=scoring,)
    validation_score = float(cv_scores.mean())
    pipeline.fit(data["features"],data["label"]) #fit the pipeline on the whole dataset
    return pipeline, validation_score