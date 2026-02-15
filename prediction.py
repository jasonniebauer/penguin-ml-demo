import joblib


MODEL_CLASSIFIER = joblib.load('penguin_classifier_model.sav')


def predict(data):
    return MODEL_CLASSIFIER.predict(data)

def predict_probability(data):
    return MODEL_CLASSIFIER.predict_proba(data)