import joblib
import pandas as pd
from datetime import datetime
from schemas import HousePredictionRequest, PredictionResponse

# Load model and preprocessor
MODEL_PATH = "models/trained/house_price_model.pkl"
PREPROCESSOR_PATH = "models/trained/preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or preprocessor: {e}")

def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    input_data = pd.DataFrame([request.dict()])
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0 
    
    # Preprocess input data
    processed_features = preprocessor.transform(input_data)
    
    # Make prediction
    predicted_price = model.predict(processed_features)[0]
    
    predicted_price = round(float(predicted_price), 2)
    
    # Confidence interval (10% range)
    confidence_interval = [predicted_price * 0.9, predicted_price * 1.1]
    
    # Convert confidence interval values to Python float and round to 2 decimal places
    confidence_interval = [round(float(value), 2) for value in confidence_interval]
    
    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        features_importance={},
        prediction_time=datetime.now().isoformat()
    )
    
def batch_predict(requests: list[HousePredictionRequest]) -> list[float]:
    """
    Perform batch predictions.
    """
    input_data = pd.DataFrame([req.dict() for req in requests])
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0  # Dummy value for compatibility

    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make predictions
    predictions = model.predict(processed_features)
    return predictions.tolist()
    
    