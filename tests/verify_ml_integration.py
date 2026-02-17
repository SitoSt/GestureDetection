import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.modules.gestures import GestureProcessor

def test_integration():
    print("Initializing GestureProcessor...")
    processor = GestureProcessor()
    
    # Load a sample file
    # Ensure this file exists or pick another one found in previous steps
    sample_file = "training_data/play_pause_INTENCIONAL/play_pause_INTENCIONAL_1762359487.json"
    
    if not os.path.exists(sample_file):
        print(f"Sample file not found: {sample_file}")
        # Try to find any json file in training_data
        for root, dirs, files in os.walk("training_data"):
             for file in files:
                 if file.endswith(".json"):
                     sample_file = os.path.join(root, file)
                     break
             if sample_file != "training_data/play_pause_INTENCIONAL/play_pause_INTENCIONAL_1762359487.json":
                 break
    
    print(f"Testing with file: {sample_file}")
    
    with open(sample_file, 'r') as f:
        data = json.load(f)
        
    # Handle dict structure
    if isinstance(data, dict) and 'sequence' in data:
        sequence = data['sequence']
    elif isinstance(data, list):
        sequence = data
    else:
        print("Invalid data format")
        return

    print(f"Sequence length: {len(sequence)}")
    
    detected = False
    for i, frame in enumerate(sequence):
        # Frame is likely: { "hands": [...], "pose": [...] }
        # Check if frame needs restructuring if it was stored differently
        # The training data seems to match expected input for .process_landmarks
        
        result = processor.process_landmarks(frame)
        
        if result:
            print(f"Frame {i}: Detected gesture -> {result}")
            detected = True
            if "play_pause" in result:
                print("SUCCESS: Correctly detected play_pause!")
                break
    
    if not detected:
        print("FAILURE: No gesture detected in the sequence.")
    else:
        print("Integration Test Passed (Gesture Detected)")

if __name__ == "__main__":
    test_integration()
