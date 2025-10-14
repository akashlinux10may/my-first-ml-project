import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_model():
    """Train a simple ML model"""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved")
    return model

def load_model():
    """Load trained model"""
    with open('models/model.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    train_model()