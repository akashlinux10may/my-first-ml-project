def preprocess_data(data):
    """Preprocess input data"""
    return data.astype(float)

def validate_input(data):
    """Validate input data shape"""
    if len(data.shape) != 2:
        raise ValueError("Data must be 2D")
    return True

def normalize_data(data):
    """Normalize data to 0-1 range"""
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)