import os
import json
import numpy as np
import sys
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder

# Add project root to path to import shared config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from shared.config import BUFFER_SIZE

TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../training_data')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../data')
MODELS_PATH = os.path.join(os.path.dirname(__file__), '../models')

# Constants for normalization
HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
TOTAL_FEATURES = (HAND_LANDMARKS * 3) + (POSE_LANDMARKS * 3)

def load_data(data_path: str) -> Tuple[List[List[Dict]], List[str]]:
    """Recursively load JSON files and assign labels based on folder names."""
    sequences = []
    labels = []
    
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".json"):
                label = os.path.basename(root)
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                        # Handle dictionary structure (new format)
                        if isinstance(data, dict) and "sequence" in data:
                            sequences.append(data["sequence"])
                            labels.append(label)
                        # Handle list structure (legacy format)
                        elif isinstance(data, list):
                            sequences.append(data)
                            labels.append(label)
                        else:
                            print(f"Skipping {file}: Invalid structure (expected dict with 'sequence' or list)")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    
    return sequences, labels

def normalize_frame(frame: Dict) -> np.ndarray:
    """Normalize a single frame of landmarks."""
    # Initialize empty arrays
    hand_vector = np.zeros(HAND_LANDMARKS * 3)
    pose_vector = np.zeros(POSE_LANDMARKS * 3)
    
    # 1. Hand Normalization
    if 'hands' in frame and len(frame['hands']) == HAND_LANDMARKS * 3:
        hands = np.array(frame['hands']).reshape(-1, 3)
        wrist = hands[0]
        
        # Center around wrist
        rel_hands = hands - wrist
        
        # Scale by max distance to wrist
        max_dist = np.max(np.linalg.norm(rel_hands, axis=1))
        if max_dist > 0:
            rel_hands = rel_hands / max_dist
            
        hand_vector = rel_hands.flatten()
        
    # 2. Pose Normalization
    if 'pose' in frame and len(frame['pose']) == POSE_LANDMARKS * 3:
        pose = np.array(frame['pose']).reshape(-1, 3)
        
        # Midpoint of shoulders (11 and 12)
        shoulder_left = pose[11]
        shoulder_right = pose[12]
        midpoint = (shoulder_left + shoulder_right) / 2
        
        # Center
        rel_pose = pose - midpoint
        
        # Scale by shoulder width
        shoulder_dist = np.linalg.norm(shoulder_left - shoulder_right)
        if shoulder_dist > 0:
            rel_pose = rel_pose / shoulder_dist
            
        pose_vector = rel_pose.flatten()
        
    return np.concatenate([hand_vector, pose_vector])

def create_sequences(raw_sequences: List[List[Dict]], raw_labels: List[str]) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Process raw sequences into sliding windows of features."""
    X = []
    y = []
    
    stride = 5  # Sliding window step
    
    for seq, label in zip(raw_sequences, raw_labels):
        # Normalize every frame in the sequence first
        normalized_seq = [normalize_frame(frame) for frame in seq]
        
        # Sliding window
        if len(normalized_seq) < BUFFER_SIZE:
             # Skip or pad? Decision: Pad with zeros/last frame or skip. 
             # Given 45 frames per file, strict skip for < 20 is safe.
             continue
             
        for i in range(0, len(normalized_seq) - BUFFER_SIZE + 1, stride):
            window = normalized_seq[i : i + BUFFER_SIZE]
            X.append(window)
            y.append(label)
            
    X = np.array(X)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    print(f"Loading data from {TRAINING_DATA_PATH}...")
    sequences, labels = load_data(TRAINING_DATA_PATH)
    print(f"Found {len(sequences)} raw sequences.")
    
    print("Processing and normalizing...")
    X, y, le = create_sequences(sequences, labels)
    print(f"Generated {len(X)} samples with shape {X.shape}")
    
    # Save filenames
    np.save(os.path.join(OUTPUT_PATH, 'X_train.npy'), X)
    np.save(os.path.join(OUTPUT_PATH, 'y_train.npy'), y)
    
    # Save label map
    label_map = {int(k): v for k, v in enumerate(le.classes_)}
    with open(os.path.join(OUTPUT_PATH, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=4)
        
    print(f"Data saved to {OUTPUT_PATH}")
    print(f"Classes: {label_map}")

if __name__ == "__main__":
    main()
