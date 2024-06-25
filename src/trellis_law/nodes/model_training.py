import json
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"Model: {type(model.named_steps['clf']).__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    return metrics

def train_models(data):
    if data.empty:
        raise ValueError("The input dataset is empty. Ensure that the data preprocessing step has run correctly and data is available.")

    X = data['content']
    y = data['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Naive Bayes': Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())]),
        'Logistic Regression': Pipeline([('vect', TfidfVectorizer()), ('clf', LogisticRegression(max_iter=1000))]),
        'SVM': Pipeline([('vect', TfidfVectorizer()), ('clf', SVC(kernel='linear'))])
    }

    performance_metrics = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        metrics = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        performance_metrics[model_name] = metrics

    # Save performance metrics to a JSON file
    metrics_filepath = './data/08_reporting/model_performance_metrics.json'
    os.makedirs(os.path.dirname(metrics_filepath), exist_ok=True)
    with open(metrics_filepath, 'w') as f:
        json.dump(performance_metrics, f, indent=4)

    # Save the best model based on F1 score
    best_model_name = max(performance_metrics, key=lambda k: performance_metrics[k]['f1_score'])
    best_model = models[best_model_name]
    joblib.dump(best_model, './data/06_models/document_classifier.pkl')

    print(f"Best model: {best_model_name} saved to './data/06_models/document_classifier.pkl'")
    
    return best_model
