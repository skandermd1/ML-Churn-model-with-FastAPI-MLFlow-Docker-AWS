from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import ChurnPredictor


app=FastAPI(title="Churn Prediction API",
            version="1.0.0" )



@app.get("/")
def root():
    return {"status":"ok"}

class CustomerData(BaseModel):
    """
    Customer data schema for churn prediction.
    
    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """
    # Demographics
    gender: str                # "Male" or "Female"
    Partner: str               # "Yes" or "No" - has partner
    Dependents: str            # "Yes" or "No" - has dependents
    
    # Phone services
    PhoneService: str          # "Yes" or "No"
    MultipleLines: str         # "Yes", "No", or "No phone service"
    
    # Internet services  
    InternetService: str       # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str        # "Yes", "No", or "No internet service"
    OnlineBackup: str          # "Yes", "No", or "No internet service"
    DeviceProtection: str      # "Yes", "No", or "No internet service"
    TechSupport: str           # "Yes", "No", or "No internet service"
    StreamingTV: str           # "Yes", "No", or "No internet service"
    StreamingMovies: str       # "Yes", "No", or "No internet service"
    
    # Account information
    Contract: str              # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str      # "Yes" or "No"
    PaymentMethod: str         # "Electronic check", "Mailed check", etc.
    
    # Numeric features
    tenure: int                # Number of months with company
    MonthlyCharges: float      # Monthly charges in dollars
    TotalCharges: float        # Total charges to date
MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "churn_model.joblib"
predictor = ChurnPredictor(MODEL_PATH)

@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.
    
    This endpoint:
    1. Receives validated customer data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns churn prediction in JSON format
    
    Expected Response:
    - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        # The predictor expects one customer record in a DataFrame.
        customer = pd.DataFrame([data.model_dump()])
        result = predictor.predict(customer).iloc[0]
        return {
            "prediction": result["churn_label"],
            "churn_probability": float(result["churn_probability"]),
            "churn_prediction": int(result["churn_prediction"]),
        }
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}

