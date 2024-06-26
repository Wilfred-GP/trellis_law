import spacy
import numpy as np

# Load SpaCy's pre-trained model
nlp = spacy.load('en_core_web_md')

def embed_texts(texts):
    embeddings = []
    for text in texts:
        doc = nlp(text)
        embeddings.append(doc.vector)
    return np.array(embeddings)


def predict_with_threshold(model, X, threshold=0.5):
    # Predict probabilities
    probs = model.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    
    # If max probability is below threshold, classify as 'other'
    preds = np.where(max_probs < threshold, 'Other', preds)
    
    return preds
