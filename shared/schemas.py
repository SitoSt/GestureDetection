#type: ignore
import json
import time
from typing import Optional, Dict, Any

PROTOCOL_VERSION = "1.0"


def serialize_landmarks(
    results,
    client_id: str = "client",
    frame_b64: Optional[str] = None,
) -> Optional[str]:
    """
    Convert MediaPipe landmark results to JSON string.

    Args:
        results: MediaPipe results object with hand and pose landmarks
        client_id: Stable identifier for the client instance
        frame_b64: Optional base64-encoded frame for future use

    Returns:
        JSON string with landmarks data, or None if no landmarks detected
    """
    data: Dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "timestamp": time.time(),
        "client_id": client_id,
    }
    has_payload = False

    if results.multi_hand_landmarks:
        lm_list = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            lm_list.extend([landmark.x, landmark.y, landmark.z])
        data["hands"] = lm_list
        has_payload = True

    if results.pose_landmarks:
        data["pose"] = [
            val for lm in results.pose_landmarks.landmark
            for val in (lm.x, lm.y, lm.z)
        ]
        has_payload = True

    if frame_b64:
        data["frame"] = frame_b64

    return json.dumps(data) if has_payload else None


def parse_message(message: str) -> Dict[str, Any]:
    """Parse client payload and enforce protocol version."""
    data = json.loads(message)
    if data.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {data.get('version')}")
    return data


def create_command_json(gesture: str) -> str:
    """Create JSON command to send from server to client."""
    return json.dumps({"gesture": gesture})
