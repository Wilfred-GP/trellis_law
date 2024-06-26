# src/document_classification/api.py

from fastapi import FastAPI, HTTPException, Request
from .nodes.utils import predict_with_threshold, parse_input_text
import joblib

app = FastAPI()

# Load the model
model = joblib.load('./data/06_models/document_classifier.pkl')

# Define API endpoint for plain text input
@app.post("/classify_document")
async def classify_document(request: Request):
    try:
        raw_text = await request.body()
        raw_text = raw_text.decode('utf-8')  # Decode bytes to string
        
        # Preprocess the document text
        parsed_text = parse_input_text(raw_text)
        
        # Predict with threshold
        category = predict_with_threshold(model, [parsed_text])[0]
        
        return {"message": "Classification successful", "label": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
