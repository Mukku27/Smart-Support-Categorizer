import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict

# Import necessary components from training script
from model_training import ( 
    load_pytorch_model, 
    predict_ticket_pytorch
)

# FastAPI app configuration
app = FastAPI(
    title="Smart Support Ticket Categorizer API",
    description="API to classify support tickets into Billing, Technical, or Other.",
    version="1.0.0"
)

# Request and response models
class TicketRequest(BaseModel):
    text: str = Field(..., min_length=10, example="My computer crashes frequently and won't boot properly")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., example="Technical")
    confidence: float = Field(..., example=0.9686)
    probabilities: Dict[str, float] = Field(..., example={
        "Technical": 0.9686,
        "Billing": 0.0181,
        "Other": 0.0133
    })

# Global pipeline storage
prediction_pipeline = {}

@app.on_event("startup")
async def startup_event():
    """Load ML model when API starts"""
    print("--> Loading ML model and pipeline components...")
    try:
        model_path = 'model/ensemble_model.pth'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model components
        model, vectorizer, label_map = load_pytorch_model(model_path)
        
        # Store in global pipeline
        prediction_pipeline['model'] = model
        prediction_pipeline['vectorizer'] = vectorizer
        prediction_pipeline['label_map'] = label_map
        prediction_pipeline['device'] = device
        
        print(f"--> ML model and pipeline loaded successfully from '{model_path}'.")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{model_path}'.")
        print("Please ensure you have run 'python train_model.py' first.")
        prediction_pipeline['model'] = None
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        prediction_pipeline['model'] = None

@app.get("/", tags=["Health Check"])
async def root():
    """Root endpoint for health checks"""
    return {"message": "Smart Support Categorizer API is running."}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: TicketRequest):
    """Accept support ticket text and return classification prediction"""
    # Check if model is available
    if not prediction_pipeline.get('model'):
        raise HTTPException(
            status_code=503, 
            detail="Model is not available. Please check server logs."
        )

    try:
        # Make prediction using loaded model
        result = predict_ticket_pytorch(
            ensemble_model=prediction_pipeline['model'],
            vectorizer=prediction_pipeline['vectorizer'], 
            label_map=prediction_pipeline['label_map'], 
            ticket_text=request.text, 
            device=prediction_pipeline['device']
        )
        
        # Validate prediction result
        if not result or 'prediction' not in result:
             raise HTTPException(
                status_code=500,
                detail="Prediction failed. Could not compute result."
            )
            
        return result

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred during prediction: {e}"
        )
