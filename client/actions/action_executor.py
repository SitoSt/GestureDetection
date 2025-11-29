"""
Action executor for gesture-triggered commands.
In a production environment, this would integrate with OS-level APIs
for media control (e.g., keyboard module, AppleScript, or D-Bus).
"""

ACTION_MAP = {
    "play_pause": "⏯️  Play/Pause",
    "next_track": "⏭️  Next Track",
    "prev_track": "⏮️  Previous Track",
    "volume_up": "🔊 Volume Up",
    "volume_down": "🔉 Volume Down",
    "open_spotify": "🎵 Open Spotify"
}

def execute_action(gesture: str) -> None:
    """Execute a local action based on the detected gesture."""
    action_description = ACTION_MAP.get(gesture)
    
    if action_description:
        print(f"[Action] {action_description} (gesture: {gesture})")
    else:
        print(f"[Action] Unknown gesture: {gesture}")
