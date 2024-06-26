import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import joblib
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from .utils import embed_texts  # Import the embed_texts function


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a machine learning model.

    Args:
        model: The machine learning model to train and evaluate.
        X_train: Training features.
        X_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.

    Returns:
        metrics: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f"Model: {type(model).__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    return metrics


def train_lstm_model(X_train, X_test, y_train, y_test, max_words=5000, max_len=100):
    """
    Train and evaluate an LSTM model.

    Args:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
        max_words: Maximum number of words to consider in the tokenizer.
        max_len: Maximum sequence length for padding.

    Returns:
        metrics: A dictionary containing accuracy, precision, recall, and F1 score.
        model: The trained LSTM model.
    """
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    num_classes = len(np.unique(y_train))
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train_enc, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Decode labels back to original categories
    y_pred = label_encoder.inverse_transform(y_pred)
    y_test_enc = label_encoder.inverse_transform(y_test_enc)

    accuracy = accuracy_score(y_test_enc, y_pred)
    precision = precision_score(y_test_enc, y_pred, average='weighted')
    recall = recall_score(y_test_enc, y_pred, average='weighted')
    f1 = f1_score(y_test_enc, y_pred, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f"LSTM Model")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test_enc, y_pred))

    return metrics, model


def train_xgboost_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate an XGBoost model.

    Args:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.

    Returns:
        metrics: A dictionary containing accuracy, precision, recall, and F1 score.
        model: The trained XGBoost model.
    """
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    model = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', xgb.XGBClassifier())
    ])
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)

    # Decode labels back to original categories
    y_pred = label_encoder.inverse_transform(y_pred)
    y_test_enc = label_encoder.inverse_transform(y_test_enc)

    accuracy = accuracy_score(y_test_enc, y_pred)
    precision = precision_score(y_test_enc, y_pred, average='weighted')
    recall = recall_score(y_test_enc, y_pred, average='weighted')
    f1 = f1_score(y_test_enc, y_pred, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    print(f"XGBoost Model")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test_enc, y_pred))

    return metrics, model


def train_models(data):
    """
    Train and evaluate multiple models, then save the best model based on F1 score.

    Args:
        data: DataFrame containing the dataset.

    Returns:
        best_model: The best performing model.
    """
    if data.empty:
        raise ValueError("The input dataset is empty. Ensure that the data preprocessing step has run correctly and data is available.")

    # Exclude "other" category
    data = data[data['category'] != 'other']

    X = data['content']
    y = data['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get Spacy embeddings for LSTM and XGBoost
    X_train_embed = embed_texts(X_train)
    X_test_embed = embed_texts(X_test)

    # Reshape embeddings for LSTM input
    X_train_embed_lstm = X_train_embed.reshape((X_train_embed.shape[0], 1, X_train_embed.shape[1]))
    X_test_embed_lstm = X_test_embed.reshape((X_test_embed.shape[0], 1, X_test_embed.shape[1]))

    models = {
        'Naive Bayes': Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())]),
        'Logistic Regression': Pipeline([('vect', TfidfVectorizer()), ('clf', LogisticRegression(max_iter=1000))]),
        'SVM': Pipeline([('vect', TfidfVectorizer()), ('clf', SVC(kernel='linear', probability=True))]),
        'Gradient Boosting': Pipeline([('vect', TfidfVectorizer()), ('clf', GradientBoostingClassifier())]),
    }

    performance_metrics = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        metrics = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        performance_metrics[model_name] = metrics

    # Train LSTM model
    lstm_metrics, lstm_model = train_lstm_model(X_train_embed_lstm, X_test_embed_lstm, y_train, y_test)
    performance_metrics['LSTM'] = lstm_metrics

    # Train XGBoost model
    xgboost_metrics, xgboost_model = train_xgboost_model(X_train, X_test, y_train, y_test)
    performance_metrics['XGBoost'] = xgboost_metrics
    
    # Save performance metrics to a JSON file
    metrics_filepath = './data/08_reporting/model_performance_metrics.json'
    os.makedirs(os.path.dirname(metrics_filepath), exist_ok=True)
    with open(metrics_filepath, 'w') as f:
        json.dump(performance_metrics, f, indent=4)

    # Save the best model based on F1 score
    best_model_name = max(performance_metrics, key=lambda k: performance_metrics[k]['f1_score'])
    if best_model_name == 'LSTM':
        best_model = lstm_model
    elif best_model_name == 'XGBoost':
        best_model = xgboost_model
    else:
        best_model = models[best_model_name]
    
    joblib.dump(best_model, './data/06_models/document_classifier.pkl')

    print(f"Best model: {best_model_name} saved to 'data/06_models/document_classifier.pkl'")
    
    return best_model
