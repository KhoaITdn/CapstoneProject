import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    model = load_model('best_model.keras')
    print(f"Input shape: {model.input_shape}")
    
    # Check if it fails with the current code's logic
    import numpy as np
    dummy_input = np.zeros((1, 48, 48, 1))
    try:
        model.predict(dummy_input)
        print("Prediction with (1, 48, 48, 1) successful.")
    except Exception as e:
        print(f"Prediction with (1, 48, 48, 1) FAILED: {e}")
        
    dummy_input_rgb = np.zeros((1, 48, 48, 3))
    try:
        model.predict(dummy_input_rgb)
        print("Prediction with (1, 48, 48, 3) successful.")
    except Exception as e:
        print(f"Prediction with (1, 48, 48, 3) FAILED: {e}")

except Exception as e:
    print(f"Error loading model: {e}")
