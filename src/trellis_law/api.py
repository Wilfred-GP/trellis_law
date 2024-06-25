# src/document_classification/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the model
model = joblib.load('./data/06_models/document_classifier.pkl')

# Define request body
class DocumentRequest(BaseModel):
    document_text: str

# Define API endpoint
@app.post("/classify_document")
def classify_document(request: DocumentRequest):
    try:
        category = model.predict([request.document_text])[0]
        return {"message": "Classification successful", "label": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
