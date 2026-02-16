import time
import random
from server.modules.gestures import GestureProcessor

def run_stress_test():
    processor = GestureProcessor()
    
    # Generate random landmarks
    # 21 points * 3 coords = 63 floats
    
    print("Starting stress test with 100,000 frames...")
    start_time = time.time()
    
    processed_count = 0
    
    for _ in range(100000):
        # Generate dummy data
        landmarks = [random.uniform(0.0, 1.0) for _ in range(63)]
        data = {'hands': landmarks}
        
        # Override cooldown to force processing every frame
        processor.last_action_time = 0 
        
        result = processor.process_landmarks(data)
        processed_count += 1
        
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Processed {processed_count} frames in {duration:.4f} seconds.")
    print(f"Average FPS: {processed_count / duration:.2f}")

if __name__ == "__main__":
    run_stress_test()
