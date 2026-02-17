"""
Action executor for gesture-triggered commands.
Integrates with OS-level APIs via pyautogui for media control.
"""
import pyautogui
import platform

# OS-specific key mapping
SYSTEM = platform.system()

if SYSTEM == "Darwin":  # macOS
    # macOS media keys are often handled differently or require specific key codes.
    # pyautogui support for media keys on macOS can be limited.
    # We use standard shortcuts where possible or specific keys.
    KEY_MAP = {
        "play_pause": "playpause",
        "next_track": "nexttrack",
        "prev_track": "prevtrack",
        "volume_up": "volumeup",
        "volume_down": "volumedown",
    }
else: # Windows / Linux
    KEY_MAP = {
        "play_pause": "playpause",
        "next_track": "nexttrack",
        "prev_track": "prevtrack",
        "volume_up": "volumeup",
        "volume_down": "volumedown",
    }

ACTION_DISPLAY_MAP = {
    "play_pause": "⏯️  Play/Pause",
    "next_track": "⏭️  Next Track",
    "prev_track": "⏮️  Previous Track",
    "volume_up": "🔊 Volume Up",
    "volume_down": "🔉 Volume Down",
    "open_spotify": "🎵 Open Spotify"
}

def execute_action(gesture: str) -> None:
    """Execute a local action based on the detected gesture."""
    action_description = ACTION_DISPLAY_MAP.get(gesture)
    key = KEY_MAP.get(gesture)
    
    if action_description:
        print(f"[Action] Executing: {action_description}")
        
        if key:
            try:
                pyautogui.press(key)
            except Exception as e:
                print(f"[Action] Error pressing key {key}: {e}")
        elif gesture == "open_spotify":
             # Example custom logic
             pass
    else:
        print(f"[Action] Unknown gesture: {gesture}")
