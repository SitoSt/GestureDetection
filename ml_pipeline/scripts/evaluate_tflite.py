import os
import numpy as np
import tensorflow as tf
import json
import random

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data')
MODELS_PATH = os.path.join(os.path.dirname(__file__), '../models')

def load_data():
    X = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
    y = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
    with open(os.path.join(DATA_PATH, 'label_map.json'), 'r') as f:
        label_map = json.load(f)
    return X, y, label_map

def main():
    model_path = os.path.join(MODELS_PATH, 'gesture_model.tflite')
    
    print(f"Loading TFLite model from {model_path}...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    X, y, label_map = load_data()
    print(f"Loaded {len(X)} samples.")
    
    # Test 10 random samples
    indices = random.sample(range(len(X)), 10)
    
    print("\n--- Running Inference Tests ---")
    correct = 0
    
    for i in indices:
        input_data = X[i:i+1].astype(np.float32)
        true_label_idx = y[i]
        true_label = label_map.get(str(true_label_idx), "Unknown")
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_idx = np.argmax(output_data)
        predicted_label = label_map.get(str(predicted_idx), "Unknown")
        confidence = output_data[0][predicted_idx]
        
        is_correct = predicted_idx == true_label_idx
        status = "✅" if is_correct else "❌"
        if is_correct:
            correct += 1
            
        print(f"{status} True: {true_label:<30} | Pred: {predicted_label:<30} | Conf: {confidence:.2f}")

    print(f"\nAccuracy on random sample: {correct}/10")

if __name__ == "__main__":
    main()
