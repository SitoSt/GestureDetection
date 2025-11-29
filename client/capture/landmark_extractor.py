#type: ignore
import mediapipe as mp
import cv2
from dataclasses import dataclass
from typing import Optional

@dataclass
class LandmarkResults:
    """Combined results from hand and pose landmark detection."""
    multi_hand_landmarks: Optional[any]
    pose_landmarks: Optional[any]

class LandmarkExtractor:
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE
        )

    def process_frame(self, frame) -> LandmarkResults:
        """Extract hand and pose landmarks from a video frame."""
        # IMPORTANT: MediaPipe expects RGB, OpenCV uses BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        
        return LandmarkResults(
            multi_hand_landmarks=hand_results.multi_hand_landmarks,
            pose_landmarks=pose_results.pose_landmarks
        )

    def draw_landmarks(self, frame, results: LandmarkResults) -> None:
        """Draw hand and pose landmarks on the frame for visualization."""
        mp_draw = mp.solutions.drawing_utils
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)