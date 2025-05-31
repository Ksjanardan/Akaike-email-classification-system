# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

# Add the project directory to the Python path to allow imports from utils and models
# This is crucial if main.py is in the root and utils.py/models.py are also in the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utils import mask_pii, demask_pii # Import your PII utilities
from models import EmailClassifier      # Import your EmailClassifier class

# Initialize FastAPI app
app = FastAPI(
    title="Email Classification and PII Masking API",
    description="API for masking PII in emails, classifying emails, and demasking PII.",
    version="1.0.0"
)

# --- Load the trained EmailClassifier model and vectorizer ---
# This part should be run once when the API starts
classifier = EmailClassifier()
model_path = "email_classifier_model.joblib"
vectorizer_path = "tfidf_vectorizer.joblib"

# Check if model files exist before attempting to load
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    try:
        classifier.load_model(model_path, vectorizer_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        # Optionally, you could raise an exception or set a flag here
        # to prevent API from starting if the model isn't loaded correctly.
        # For now, we'll just print and allow it to proceed, though prediction
        # endpoints would then fail.
else:
    print(f"Model files not found. Expected at: {model_path} and {vectorizer_path}")
    print("Please ensure you have run models.py successfully to train and save the model.")
    # Set classifier to None or raise error if model is mandatory for API functionality
    # For now, we'll let it be None, but prediction will fail if not loaded.
    classifier = None 


# Define a Pydantic model for incoming email requests
class EmailRequest(BaseModel):
    email_content: str

# Define a Pydantic model for masking results (optional for a dedicated endpoint, but good for internal structure)
class MaskingResult(BaseModel):
    masked_email: str
    masked_entities: dict # This will store the PII details for demasking

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to the Email Classification and PII Masking API!"}

@app.post("/mask-pii/")
async def mask_pii_endpoint(request: EmailRequest):
    """
    Endpoint to mask PII from an incoming email.
    """
    try:
        masked_email, masked_entities = mask_pii(request.email_content)
        return {"masked_email": masked_email, "masked_entities": masked_entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during PII masking: {e}")

@app.post("/classify/")
async def classify_email_endpoint(request: EmailRequest):
    """
    Endpoint to classify an incoming email.
    The email should ideally be masked before sending to this endpoint if not already.
    This endpoint will handle masking internally for robustness.
    """
    if classifier is None or classifier.model is None or classifier.vectorizer is None:
        raise HTTPException(status_code=500, detail="Classification model not loaded. Please ensure models.py was run successfully.")
    
    try:
        # First, mask PII from the incoming email
        masked_email_content, _ = mask_pii(request.email_content)
        
        # Then, classify the masked email
        predicted_category = classifier.predict(masked_email_content)
        
        return {
            "original_email_snippet": request.email_content[:100] + "...", # Return a snippet for context
            "masked_email_snippet": masked_email_content[:100] + "...",
            "predicted_category": predicted_category
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during email classification: {e}")

@app.post("/demask-pii/")
async def demask_pii_endpoint(data: MaskingResult):
    """
    Endpoint to demask an email using previously stored masked entities.
    Expects 'masked_email' and 'masked_entities' from a prior masking operation.
    """
    try:
        demasked_email = demask_pii(data.masked_email, data.masked_entities)
        return {"demasked_email": demasked_email}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during PII demasking: {e}")