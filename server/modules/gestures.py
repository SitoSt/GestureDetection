from collections import deque
import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from shared.config import (
    PINCH_THRESHOLD_3D, VOLUME_MOVE_THRESHOLD, FIST_DISTANCE_THRESHOLD,
    COOLDOWN, GESTURE_STABILITY_FRAMES, BUFFER_SIZE, SMOOTHING_WINDOW,
    MODEL_CONFIDENCE_THRESHOLD, INFERENCE_INTERVAL, GESTURE_THRESHOLDS
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
        self.frame_counter = 0
        
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

    def _normalize_features(self, hands_list: List[float], pose_list: List[float]) -> List[float]:
        """
        Normalize landmarks to match model input (162 features: 63 hands + 99 pose).
        Hands: Relative to wrist, scaled by max dist to wrist.
        Pose: Relative to shoulder midpoint, scaled by shoulder width.
        """
        # 1. Hands (63 features)
        if not hands_list or len(hands_list) < 21 * 3:
            hand_vector = np.zeros(21 * 3)
        else:
            try:
                hands = np.array(hands_list).reshape(-1, 3)
                wrist = hands[0]
                rel_hands = hands - wrist
                max_dist = np.max(np.linalg.norm(rel_hands, axis=1))
                if max_dist > 0:
                    rel_hands = rel_hands / max_dist
                hand_vector = rel_hands.flatten()
            except Exception:
                hand_vector = np.zeros(21 * 3)

        # 2. Pose (99 features)
        if not pose_list or len(pose_list) < 33 * 3:
            pose_vector = np.zeros(33 * 3)
        else:
            try:
                pose = np.array(pose_list).reshape(-1, 3)
                # Midpoint of shoulders (11 and 12)
                shoulder_left = pose[11]
                shoulder_right = pose[12]
                midpoint = (shoulder_left + shoulder_right) / 2
                
                rel_pose = pose - midpoint
                shoulder_dist = np.linalg.norm(shoulder_left - shoulder_right)
                if shoulder_dist > 0:
                    rel_pose = rel_pose / shoulder_dist
                pose_vector = rel_pose.flatten()
            except Exception:
                pose_vector = np.zeros(33 * 3)
                
        # Concatenate
        return np.concatenate([hand_vector, pose_vector]).tolist()

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
            # If hands are missing, we still want to process for ML buffer (as zeros)
            # preventing "None" return that blocks buffer filling.
            raw_lm_list = []
        else:
            raw_lm_list = data['hands']
        
        # 1. Smoothing
        if raw_lm_list:
             smoothed_lm_list = self._smooth_landmarks(raw_lm_list)
        else:
             smoothed_lm_list = []
        
        now = time.time()
        
        # 2. Normalization & Buffering (for ML)
        # Use smoothed hands and raw pose (pose smoothing not implemented yet)
        pose_list = data.get('pose', [])
        normalized_features = self._normalize_features(smoothed_lm_list, pose_list)
        self.landmark_buffer.append(normalized_features)
        
        # DEBUG
        # print(f"[DEBUG] Frame processing. Buffer: {len(self.landmark_buffer)}")

        # 3. Cooldown Check
        if now - self.last_action_time < COOLDOWN:
            return None

        # 4. Priority: Pinch (Volume Control) - Heuristic
        # We check this first because it's a continuous interaction
        try:
           index_tip_y = self._get_coords(smoothed_lm_list, 8)[1]
           
           # Check for pinch specifically
           thumb_tip = self._get_coords(smoothed_lm_list, 4)
           index_tip = self._get_coords(smoothed_lm_list, 8)
           dist_thumb_index = self._get_distance_3d(thumb_tip, index_tip)
           
           if dist_thumb_index < PINCH_THRESHOLD_3D:
               # Pinch detected
               self.current_stable_gesture = None
               self.gesture_count = 0
               
               volume_command = self._handle_volume(index_tip_y)
               if volume_command:
                   self.last_action_time = now
                   return volume_command
               return None
           else:
               self.last_index_y = None
               
        except IndexError:
            pass

        # 5. ML Model Prediction
        # Only predict if we have enough history AND it's the right interval
        self.frame_counter += 1
        if len(self.landmark_buffer) == BUFFER_SIZE and (self.frame_counter % INFERENCE_INTERVAL == 0):
            ml_gesture, ml_confidence = self.model.predict(list(self.landmark_buffer))
            
            if ml_gesture:
                # Dynamic Threshold Lookup
                threshold = GESTURE_THRESHOLDS.get(ml_gesture, MODEL_CONFIDENCE_THRESHOLD)
                
                # Clean gesture name for lookup if needed (e.g. handle suffixes)
                clean_name = ml_gesture.replace("_INTENCIONAL", "")
                threshold = GESTURE_THRESHOLDS.get(clean_name, threshold)

                if ml_confidence > threshold:
                    # Filter "negative" classes
                    if "NO_ACTION" in ml_gesture or "falso_positivo" in ml_gesture or "no_accion" in ml_gesture:
                        return None
                    
                    # Use clean name
                    command = clean_name
                    
                    # Return immediately
                    print(f"[GestureProcessor] ML Action: {command} ({ml_confidence:.2f} > {threshold})")
                    self.last_action_time = now
                    return command
        
        return None