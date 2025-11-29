#type: ignore
import numpy as np
import time
from typing import Tuple, Optional, Dict

class GestureProcessor:
    """
    Advanced gesture detection with 3D analysis, temporal stability, and contextual filtering.
    
    Features:
    - 3D landmark distance calculation for improved accuracy
    - Multi-frame gesture stability verification
    - Contextual validation (hand-to-face proximity detection)
    - Cooldown period to prevent rapid re-triggering
    """
    
    # Thresholds
    PINCH_THRESHOLD_3D = 0.045
    VOLUME_MOVE_THRESHOLD = 0.025
    FIST_DISTANCE_THRESHOLD = 0.1
    
    # Temporal filtering
    COOLDOWN = 0.8
    GESTURE_STABILITY_FRAMES = 5
    
    def __init__(self):
        self.last_action_time = 0
        self.last_index_y: Optional[float] = None
        self.current_stable_gesture: Optional[str] = None
        self.gesture_count = 0

    def _get_coords(self, lm_list: list, index: int) -> Tuple[float, float, float]:
        """Extract x, y, z coordinates of a specific landmark by index."""
        if len(lm_list) < (index * 3 + 2):
            return 2.0, 2.0, 0.0
        
        x = lm_list[index * 3]
        y = lm_list[index * 3 + 1]
        z = lm_list[index * 3 + 2]
        return x, y, z

    def _get_distance_3d(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculate 3D Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def _detect_raw_gesture(self, lm_list: list) -> Optional[str]:
        """
        Detect gesture based on 3D hand shape and 2D (Y-axis) finger positions.
        
        Returns:
            - "pinch_active": Thumb and index finger close together
            - "next_track_shape": Two fingers extended (index and middle)
            - "play_pause_shape": Fist gesture
            - None: No gesture detected
        """
        try:
            thumb_tip = self._get_coords(lm_list, 4)
            index_tip = self._get_coords(lm_list, 8)
            middle_tip = self._get_coords(lm_list, 12)
            ring_tip = self._get_coords(lm_list, 16)
            
            index_pip_y = self._get_coords(lm_list, 6)[1]
            middle_pip_y = self._get_coords(lm_list, 10)[1]
            ring_pip_y = self._get_coords(lm_list, 14)[1]
            pinky_pip_y = self._get_coords(lm_list, 18)[1]
            
            wrist_y = self._get_coords(lm_list, 0)[1]
        except IndexError:
            return None
        
        dist_thumb_index_3d = self._get_distance_3d(thumb_tip, index_tip)
        
        # 1. Pinch detection (volume control activation)
        if dist_thumb_index_3d < self.PINCH_THRESHOLD_3D:
            return "pinch_active"
        
        # 2. Two-finger gesture (next track)
        index_extended = index_tip[1] < index_pip_y
        middle_extended = middle_tip[1] < middle_pip_y
        ring_closed = ring_tip[1] > ring_pip_y
        pinky_closed = ring_tip[1] > pinky_pip_y
        
        if index_extended and middle_extended and ring_closed and pinky_closed:
            return "next_track_shape"
        
        # 3. Fist gesture (play/pause)
        index_closed = index_tip[1] > index_pip_y
        middle_closed = middle_tip[1] > middle_pip_y
        ring_closed_check = ring_tip[1] > ring_pip_y
        is_not_pinch = dist_thumb_index_3d > 0.1
        
        index_knuckle_z = self._get_coords(lm_list, 6)[2]
        thumb_tip_z = thumb_tip[2]
        thumb_is_tucked = index_knuckle_z < thumb_tip_z
        
        if index_closed and middle_closed and ring_closed_check and is_not_pinch and thumb_is_tucked:
            return "play_pause_shape"
        
        return None

    def _handle_volume(self, index_tip_y: float) -> Optional[str]:
        """
        Handle volume control based on vertical (Y-axis) movement of index finger.
        
        Args:
            index_tip_y: Y coordinate of index finger tip
            
        Returns:
            "volume_up", "volume_down", or None
        """
        if self.last_index_y is None:
            self.last_index_y = index_tip_y
            return None
        
        dy = self.last_index_y - index_tip_y
        
        if dy > self.VOLUME_MOVE_THRESHOLD:
            self.last_index_y = index_tip_y
            return "volume_up"
        elif dy < -self.VOLUME_MOVE_THRESHOLD:
            self.last_index_y = index_tip_y
            return "volume_down"
        
        return None

    def is_gesture_contextually_valid(self, data: Dict) -> bool:
        """
        Contextual validation to filter false positives (e.g., hand near face).
        
        This prevents play/pause detection when the fist is near the mouth/face
        (e.g., smoking, eating, scratching).
        
        Args:
            data: Dictionary containing 'hands' and optionally 'pose' landmarks
            
        Returns:
            True if gesture is valid, False if it should be filtered out
        """
        if self.current_stable_gesture != "play_pause_shape":
            return True
        
        if 'pose' not in data or 'hands' not in data:
            return True
        
        pose_lm = data['pose']
        hand_lm = data['hands']
        
        try:
            nose_y = pose_lm[1]
            nose_x = pose_lm[0]
            
            wrist_x, wrist_y, _ = self._get_coords(hand_lm, 0)
            
            # Filter out if wrist is too close to face
            if abs(wrist_y - nose_y) < 0.25 and abs(wrist_x - nose_x) < 0.25:
                return False
        except IndexError:
            return True
        
        return True

    def process_landmarks(self, data: Dict) -> Optional[str]:
        """
        Main processing function with filtering, stability, and contextual validation.
        
        Processing pipeline:
        1. Cooldown check (prevent rapid re-triggering)
        2. Raw gesture detection (shape analysis)
        3. Pinch handling (volume control)
        4. Stability verification (multi-frame confirmation)
        5. Contextual validation (false positive filtering)
        
        Args:
            data: Dictionary with 'hands' and optionally 'pose' landmarks
            
        Returns:
            Gesture command string or None
        """
        now = time.time()
        
        if now - self.last_action_time < self.COOLDOWN:
            return None
        
        if 'hands' not in data:
            self.current_stable_gesture = None
            self.gesture_count = 0
            self.last_index_y = None
            return None
        
        lm_list = data['hands']
        index_tip_y = self._get_coords(lm_list, 8)[1]
        
        raw_gesture = self._detect_raw_gesture(lm_list)
        
        # Handle pinch (volume control)
        if raw_gesture == "pinch_active":
            self.current_stable_gesture = None
            self.gesture_count = 0
            
            volume_command = self._handle_volume(index_tip_y)
            if volume_command:
                self.last_action_time = now
                return volume_command
            return None
        else:
            self.last_index_y = None
        
        # Stability logic for shape gestures
        is_shape_gesture = raw_gesture in ["play_pause_shape", "next_track_shape"]
        
        if is_shape_gesture and raw_gesture == self.current_stable_gesture:
            self.gesture_count += 1
        elif is_shape_gesture and raw_gesture != self.current_stable_gesture:
            self.current_stable_gesture = raw_gesture
            self.gesture_count = 1
            return None
        elif raw_gesture is None:
            self.current_stable_gesture = None
            self.gesture_count = 0
        
        # Confirmation by stability and context
        if self.gesture_count >= self.GESTURE_STABILITY_FRAMES:
            if self.is_gesture_contextually_valid(data):
                command = self.current_stable_gesture.replace("_shape", "")
                self.last_action_time = now
                self.gesture_count = 0
                self.current_stable_gesture = None
                
                print(f"[GestureProcessor] Confirmed gesture: {command}")
                return command
            else:
                print("[GestureProcessor] False positive filtered by contextual logic")
                self.gesture_count = 0
                self.current_stable_gesture = None
                return None
        
        return None