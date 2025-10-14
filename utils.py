def preprocess_data(data):
    """Preprocess input data"""
    return data.astype(float)

def validate_input(data):
    """Validate input data shape"""
    if len(data.shape) != 2:
        raise ValueError("Data must be 2D")
    return True