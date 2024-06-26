import spacy
import numpy as np
from .data_engineering import preprocess_text

# Load SpaCy's pre-trained model
nlp = spacy.load('en_core_web_md')

def embed_texts(texts: list) -> np.ndarray:
    """
    Converts a list of texts into their corresponding embeddings using SpaCy's pre-trained model.

    Args:
        texts: A list of texts to be embedded.

    Returns:
        A numpy array containing the embeddings of the texts.
    """
    embeddings = []
    for text in texts:
        doc = nlp(text)
        embeddings.append(doc.vector)
    return np.array(embeddings)

def parse_input_text(text: str) -> str:
    """
    Preprocesses input text using the preprocess_text function from data_engineering module.

    Args:
        text: The text to preprocess.

    Returns:
        The preprocessed text.
    """
    return preprocess_text(text)

def predict_with_threshold(model, X: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    """
    Predicts the class of input samples and applies a threshold to classify uncertain samples as 'Other'.

    Args:
        model: The trained classification model.
        X: The input features.
        threshold: The probability threshold below which samples are classified as 'Other'.

    Returns:
        An array of predicted classes with 'Other' for samples below the threshold.
    """
    # Predict probabilities
    probs = model.predict_proba(X)
    category = model.predict(X)
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    
    # If max probability is below threshold, classify as 'Other'
    preds = np.where(max_probs < threshold, 'Other', category)
    
    return preds
