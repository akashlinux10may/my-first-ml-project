import numpy as np
from model_training import load_model
import logging

logging.basicConfig(level=logging.INFO)

def predict(data):
    """Make predictions using the trained model"""
    model = load_model()
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    print("ML Pipeline initialized")
    sample_data = np.array([[1, 2, 3]])
    result = predict(sample_data)
    print(f"Prediction result: {result}")