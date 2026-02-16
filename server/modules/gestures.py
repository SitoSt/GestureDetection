from collections import deque
import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from shared.config import (
    PINCH_THRESHOLD_3D, VOLUME_MOVE_THRESHOLD, FIST_DISTANCE_THRESHOLD,
    COOLDOWN, GESTURE_STABILITY_FRAMES, BUFFER_SIZE, SMOOTHING_WINDOW
)
from server.modules.model_loader import GestureModel

class GestureProcessor:
    """
    Advanced gesture detection with 3D analysis, temporal stability, and contextual filtering.
    Now supports normalization, smoothing, and ML-readiness.
    """
    
    def __init__(self):
        self.last_action_time = 0
        self.last_index_y: Optional[float] = None
        self.current_stable_gesture: Optional[str] = None
        self.gesture_count = 0
        
        # ML & Data Processing
        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.landmark_buffer = deque(maxlen=BUFFER_SIZE)
        self.model = GestureModel()

    def _get_coords(self, lm_list: list, index: int) -> Tuple[float, float, float]:
        """Extract x, y, z coordinates of a specific landmark by index."""
        if len(lm_list) < (index * 3 + 2):
            return 0.0, 0.0, 0.0
        
        x = lm_list[index * 3]
        y = lm_list[index * 3 + 1]
        z = lm_list[index * 3 + 2]
        return x, y, z

    def _get_distance_3d(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculate 3D Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def _smooth_landmarks(self, lm_list: List[float]) -> List[float]:
        """Apply moving average smoothing to landmarks."""
        self.history.append(lm_list)
        
        if not self.history:
            return lm_list
            
        # Compute mean across history for each coordinate
        # history is [ [x1, y1, z1, ...], [x2, y2, z2, ...], ... ]
        smoothed = np.mean(self.history, axis=0).tolist()
        return smoothed

    def _normalize_landmarks(self, lm_list: List[float]) -> List[float]:
        """
        Normalize landmarks: relative to wrist (index 0) and scaled by hand size.
        """
        if not lm_list or len(lm_list) < 3:
            return []

        # Get wrist coordinates
        wrist_x, wrist_y, wrist_z = lm_list[0], lm_list[1], lm_list[2]
        
        normalized = []
        max_dist = 0.0
        
        # Shift relative to wrist and find max distance for scaling
        temp_points = []
        for i in range(0, len(lm_list), 3):
            if i + 2 >= len(lm_list):
                break
            
            x, y, z = lm_list[i], lm_list[i+1], lm_list[i+2]
            dx, dy, dz = x - wrist_x, y - wrist_y, z - wrist_z
            
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > max_dist:
                max_dist = dist
            
            temp_points.append((dx, dy, dz))
            
        # Scale
        scale = max_dist if max_dist > 0 else 1.0
        
        for dx, dy, dz in temp_points:
            normalized.extend([dx / scale, dy / scale, dz / scale])
            
        return normalized

    def _detect_raw_gesture(self, lm_list: list) -> Optional[str]:
        """
        Detect gesture based on 3D hand shape and 2D (Y-axis) finger positions.
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
            
            # Unused currently but kept for structure
            # wrist_y = self._get_coords(lm_list, 0)[1]
        except IndexError:
            return None
        
        dist_thumb_index_3d = self._get_distance_3d(thumb_tip, index_tip)
        
        # 1. Pinch detection (volume control activation)
        if dist_thumb_index_3d < PINCH_THRESHOLD_3D:
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
        """
        if self.last_index_y is None:
            self.last_index_y = index_tip_y
            return None
        
        dy = self.last_index_y - index_tip_y
        
        if dy > VOLUME_MOVE_THRESHOLD:
            self.last_index_y = index_tip_y
            return "volume_up"
        elif dy < -VOLUME_MOVE_THRESHOLD:
            self.last_index_y = index_tip_y
            return "volume_down"
        
        return None

    def is_gesture_contextually_valid(self, data: Dict, wrist_coords: Tuple[float, float, float]) -> bool:
        """
        Contextual validation to filter false positives (e.g., hand near face).
        Requires original data for pose, and smoothed wrist coords.
        """
        if self.current_stable_gesture != "play_pause_shape":
            return True
        
        if 'pose' not in data:
            return True
        
        pose_lm = data['pose']
        
        try:
            # Pose landmarks 0 is nose
            nose_x = pose_lm[0]
            nose_y = pose_lm[1]
            
            wrist_x, wrist_y, _ = wrist_coords
            
            # Filter out if wrist is too close to face
            if abs(wrist_y - nose_y) < 0.25 and abs(wrist_x - nose_x) < 0.25:
                return False
        except IndexError:
            return True
        
        return True

    def process_landmarks(self, data: Dict) -> Optional[str]:
        """
        Main processing function with filtering, stability, and contextual validation.
        """
        if 'hands' not in data:
            self.current_stable_gesture = None
            self.gesture_count = 0
            self.last_index_y = None
            return None
        
        now = time.time()
        
        raw_lm_list = data['hands']
        
        # 1. Smoothing
        smoothed_lm_list = self._smooth_landmarks(raw_lm_list)
        
        # 2. Normalization & Buffering (for ML)
        normalized_lm = self._normalize_landmarks(smoothed_lm_list)
        self.landmark_buffer.append(normalized_lm)
        
        # 3. Cooldown Check
        if now - self.last_action_time < COOLDOWN:
            return None
        
        # 3. Model Prediction (Placeholder)
        ml_gesture, ml_confidence = self.model.predict(list(self.landmark_buffer))
        if ml_gesture and ml_confidence > 0.8:
             print(f"[GestureProcessor] ML Predicted: {ml_gesture} ({ml_confidence:.2f})")
             # Could return ml_gesture here in the future
        
        # 4. Legacy Heuristic Detection (using smoothed landmarks)
        lm_list = smoothed_lm_list # Use smoothed for heuristics
        
        try:
           index_tip_y = self._get_coords(lm_list, 8)[1]
        except:
           return None

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
        if self.gesture_count >= GESTURE_STABILITY_FRAMES:
            wrist_coords = self._get_coords(lm_list, 0)
            if self.is_gesture_contextually_valid(data, wrist_coords):
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