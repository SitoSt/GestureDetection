import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from sklearn.model_selection import train_test_split

# Add project root to path to import shared config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from shared.config import BUFFER_SIZE

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data')
MODELS_PATH = os.path.join(os.path.dirname(__file__), '../models')

def load_data():
    """Load preprocessed data."""
    X = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
    y = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
    return X, y

def build_model(input_shape, num_classes):
    """Build GRU model."""
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=False, unroll=True),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    print("Loading data...")
    try:
        X, y = load_data()
    except FileNotFoundError:
        print("Data not found. Run preprocess.py first.")
        return

    num_classes = len(np.unique(y))
    input_shape = (X.shape[1], X.shape[2]) # (BUFFER_SIZE, Features)
    
    print(f"Input Shape: {input_shape}")
    print(f"Num Classes: {num_classes}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building model...")
    model = build_model(input_shape, num_classes)
    model.summary()
    
    print("Training model...")
    history = model.fit(X_train, y_train, 
                        epochs=30, 
                        batch_size=32, 
                        validation_data=(X_val, y_val))
    
    print("Evaluating model...")
    loss, acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {acc*100:.2f}%")
    
    # Save standard model
    model.save(os.path.join(MODELS_PATH, 'gesture_model.keras'))
    
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization (optional but recommended for mobile/edge)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(MODELS_PATH, 'gesture_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    main()
