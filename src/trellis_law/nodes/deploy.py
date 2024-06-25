import joblib

def save_model(model):
    joblib.dump(model, './data/06_models/document_classifier.pkl')
