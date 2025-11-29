#type: ignore
import json
from typing import Optional, Dict, Any

def serialize_landmarks(results, frame_b64: Optional[str] = None) -> Optional[str]:
    """
    Convert MediaPipe landmark results to JSON string.
    
    Args:
        results: MediaPipe results object with hand and pose landmarks
        frame_b64: Optional base64-encoded frame for future use
        
    Returns:
        JSON string with landmarks data, or None if no landmarks detected
    """
    data: Dict[str, Any] = {}

    if results.multi_hand_landmarks:
        lm_list = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            lm_list.extend([landmark.x, landmark.y, landmark.z])
        data['hands'] = lm_list

    if results.pose_landmarks:
        data['pose'] = [
            val for lm in results.pose_landmarks.landmark 
            for val in (lm.x, lm.y, lm.z)
        ]
    
    if frame_b64:
        data['frame'] = frame_b64
        
    return json.dumps(data) if data else None

def create_command_json(gesture: str) -> str:
    """Create JSON command to send from server to client."""
    return json.dumps({"gesture": gesture})