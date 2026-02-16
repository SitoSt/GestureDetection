import unittest
import numpy as np
from collections import deque
from server.modules.gestures import GestureProcessor
from shared.config import BUFFER_SIZE, SMOOTHING_WINDOW

class TestGestureProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = GestureProcessor()

    def test_initialization(self):
        self.assertEqual(len(self.processor.history), 0)
        self.assertEqual(len(self.processor.landmark_buffer), 0)
        self.assertEqual(self.processor.history.maxlen, SMOOTHING_WINDOW)
        self.assertEqual(self.processor.landmark_buffer.maxlen, BUFFER_SIZE)

    def test_normalization(self):
        # Create a simple hand: wrist at (10, 10, 10), index tip at (12, 12, 12)
        # Distance = sqrt(2^2 + 2^2 + 2^2) = sqrt(12) approx 3.46
        # Normalized wrist should be (0, 0, 0)
        # Normalized index tip should be (2/3.46, 2/3.46, 2/3.46) approx (0.577, ...)
        
        # Mock landmarks: 21 points * 3 coords
        # Let's make all points same as wrist except index tip (index 8)
        landmarks = [10.0] * 63
        landmarks[0], landmarks[1], landmarks[2] = 10.0, 10.0, 10.0 # Wrist
        landmarks[8*3], landmarks[8*3+1], landmarks[8*3+2] = 14.0, 14.0, 10.0 # Index Tip at (14, 14, 10)
        
        # Dist = sqrt(4^2 + 4^2 + 0) = sqrt(32) = 5.65
        
        normalized = self.processor._normalize_landmarks(landmarks)
        
        # Check wrist is 0
        self.assertAlmostEqual(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[1], 0.0)
        self.assertAlmostEqual(normalized[2], 0.0)
        
        # Check buffer update via process_landmarks
        # We need to wrap landmarks in a dict as expected by process_landmarks
        data = {'hands': landmarks}
        self.processor.process_landmarks(data)
        
        self.assertEqual(len(self.processor.landmark_buffer), 1)
        self.assertEqual(len(self.processor.history), 1)

    def test_smoothing(self):
        # Frame 1: all 0
        lm1 = [0.0] * 63
        self.processor.process_landmarks({'hands': lm1})
        
        # Frame 2: all 1
        lm2 = [10.0] * 63
        self.processor.process_landmarks({'hands': lm2})
        
        # Frame 3: all 2
        lm3 = [20.0] * 63
        self.processor.process_landmarks({'hands': lm3})
        
        # Smoothed should be average of 0, 10, 20 -> 10
        # The history has [lm1, lm2, lm3]
        # internal _smooth_landmarks is called inside process_landmarks.
        # But we can verify by checking the history buffer directly or by mocking
        
        # Let's check history directly
        self.assertEqual(len(self.processor.history), 3)
        self.assertEqual(self.processor.history[0][0], 0.0)
        self.assertEqual(self.processor.history[2][0], 20.0)
        
        # Let's call _smooth_landmarks directly to verifying calculation
        smoothed = self.processor._smooth_landmarks([20.0]*63) # this adds a 4th element, popping the first (0)
        # Now buffer is [10, 20, 20] -> avg = 50/3 = 16.66
        self.assertAlmostEqual(smoothed[0], 16.666666, places=4)

if __name__ == '__main__':
    unittest.main()
