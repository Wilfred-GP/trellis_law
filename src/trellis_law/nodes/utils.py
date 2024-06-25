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