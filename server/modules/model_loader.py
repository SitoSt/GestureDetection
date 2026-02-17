import os
import json
try:
    # Try importing tflite_runtime first (lightweight)
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback to full TensorFlow
    import tensorflow.lite as tflite
import numpy as np
from typing import List, Optional, Tuple

class GestureModel:
    """
    ML-based gesture recognition model using TFLite.
    """
    def __init__(self):
        # Paths
        base_dir = os.path.dirname(__file__)
        # Assuming server/modules -> ../../ml_pipeline/models
        # Adjust path to match where models are actually located relative to this file
        # workspace/server/modules/model_loader.py
        # workspace/ml_pipeline/models/gesture_model.tflite
        self.model_path = os.path.abspath(os.path.join(base_dir, '../../ml_pipeline/models/gesture_model.tflite'))
        self.label_map_path = os.path.abspath(os.path.join(base_dir, '../../ml_pipeline/data/label_map.json'))
        
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.label_map = {}
        
        try:
            self._load_model()
            self._load_labels()
            print(f"[GestureModel] Successfully loaded model from {self.model_path}")
        except Exception as e:
            print(f"[GestureModel] Error loading model: {e}")
            # Fallback or just re-raise depending on strictness. 
            # For now, print error so server creates it but maybe fails on predict.

    def _load_model(self):
        from shared.config import BUFFER_SIZE
        
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Validate logic vs model
        input_shape = self.input_details[0]['shape'] # [1, 20, 162]
        model_time_steps = input_shape[1]
        
        if model_time_steps != BUFFER_SIZE:
             print(f"[GestureModel] CRITICAL WARNING: Model expects {model_time_steps} frames, but config BUFFER_SIZE is {BUFFER_SIZE}.")
             # We could raise an error here, but for now a loud warning allows debugging.

    def _load_labels(self):
        if os.path.exists(self.label_map_path):
            with open(self.label_map_path, 'r') as f:
                self.label_map = json.load(f)
        else:
            print(f"[GestureModel] Warning: Label map not found at {self.label_map_path}")

    def predict(self, landmark_buffer: List[List[float]]) -> Tuple[Optional[str], float]:
        """
        Predict gesture from a buffer of normalized landmarks.
        
        Args:
            landmark_buffer: List of normalized landmark frames (20 frames, 162 features each).
        
        Returns:
            Tuple containing:
            - Detected gesture name (or None)
            - Confidence score (0.0 to 1.0)
        """
        if not self.interpreter:
            return None, 0.0
            
        if len(landmark_buffer) != 20:
             # Buffer must be exactly BUFFER_SIZE (20)
             return None, 0.0

        try:
            # Prepare input data
            # Model expects [1, 20, 162], float32
            input_data = np.array([landmark_buffer], dtype=np.float32)
            
            # Check shape consistency
            input_shape = self.input_details[0]['shape']
            # Expected [1, 20, 162]
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Get prediction
            predicted_idx = np.argmax(output_data[0])
            confidence = float(output_data[0][predicted_idx])
            
            gesture_name = self.label_map.get(str(predicted_idx), f"Unknown_{predicted_idx}")
            
            # Clean up gesture name if it contains suffix like _INTENCIONAL
            # Or leave it raw and let logic handle it. 
            # Request says: "mapearlo al nombre del gesto".
            return gesture_name, confidence

        except Exception as e:
            print(f"[GestureModel] Inference error: {e}")
            return None, 0.0
