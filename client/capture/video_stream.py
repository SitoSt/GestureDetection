import cv2
import asyncio
from typing import Optional
from .landmark_extractor import LandmarkExtractor
from client.ws_client import WebSocketClient
from shared.schemas import serialize_landmarks

class VideoStream:
    TARGET_FPS = 30
    WINDOW_NAME = "GestureDetection Client"
    
    # HUD Colors (BGR)
    COLOR_GREY = (100, 100, 100)
    COLOR_BLUE = (255, 100, 0)   # Blue-ish
    COLOR_GREEN = (0, 255, 0)
    
    def __init__(self, ws_client: WebSocketClient):
        self.extractor = LandmarkExtractor()
        self.ws_client = ws_client
        self.cap = cv2.VideoCapture(0)
        
        # Register callback for HUD
        self.ws_client.on_command_callback = self._on_gesture_received
        
        # HUD State
        self.last_gesture = None
        self.last_gesture_time = 0
        self.display_timer = 0 # Frames to show gesture
        self.current_volume = 50 # Mock volume for display
        
    async def start_streaming(self) -> None:
        """Video capture and WebSocket communication."""
        print("[VideoStream] Starting video capture...")
        if not self.cap.isOpened():
            print("[VideoStream] Error: Cannot open camera.")
            return
        
        await asyncio.gather(
            self._capture_loop(),
            self.ws_client.connect_and_listen()
        )

    async def _capture_loop(self) -> None:
        """Main loop for split-process (capture -> extract -> send)."""
        import time
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            results = self.extractor.process_frame(frame)
            
            # --- HUD LOGIC ---
            status_color = self.COLOR_GREY
            
            has_hands = bool(results.multi_hand_landmarks)
            if has_hands:
                status_color = self.COLOR_BLUE
                
                # Serialization
                data_json = serialize_landmarks(results, client_id=self.ws_client.client_id)
                if data_json:
                    await self.ws_client.send_data(data_json)

            # Check for incoming commands (Mock logic: usually comes via ws_client callback/queue)
            # For now, we rely on the main loop or a shared state if we want to display SERVER response.
            # Assuming ws_client updates a 'last_received_command' attribute or similar.
            # But the request says client receives commands. 
            # We need to bridge ws_client message handling to this view.
            # Let's assume ws_client has a queue or callback.
            
            # Draw HUD
            self._draw_hud(frame, status_color)
            
            self.extractor.draw_landmarks(frame, results)
            cv2.imshow(self.WINDOW_NAME, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(1 / self.TARGET_FPS)

        self.cap.release()
        cv2.destroyAllWindows()
        print("[VideoStream] Video capture stopped.")

    def _on_gesture_received(self, gesture: str):
        """Callback when a gesture command is received via WebSocket."""
        self.last_gesture = gesture
        self.display_timer = 60 # Show for ~2 seconds at 30fps
        
        # Volume Mock Update
        if gesture == "volume_up":
            self.current_volume = min(100, self.current_volume + 10)
        elif gesture == "volume_down":
            self.current_volume = max(0, self.current_volume - 10)

    def _draw_hud(self, frame, status_color):
        h, w, _ = frame.shape
        import time
        
        # 1. Status Indicator (Circle in corner)
        cv2.circle(frame, (30, 30), 10, status_color, -1)
        
        # 2. Status Text
        state_text = "Idle" if status_color == self.COLOR_GREY else "Detecting"
        cv2.putText(frame, state_text, (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # 3. Gesture Feedback
        if self.display_timer > 0:
            self.display_timer -= 1
            
            # Gesture Name (Large, Centered Top)
            if self.last_gesture:
                text = self.last_gesture.replace("_", " ").upper()
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, text, (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.COLOR_GREEN, 3)
            
            # Volume Bar (if volume action)
            if self.last_gesture in ["volume_up", "volume_down"]:
                # Draw bar container
                bar_x = w - 50
                bar_y = h // 2
                bar_w = 20
                bar_h = 200
                cv2.rectangle(frame, (bar_x, bar_y - bar_h//2), (bar_x + bar_w, bar_y + bar_h//2), (200, 200, 200), 2)
                
                # Draw level
                fill_h = int((self.current_volume / 100) * bar_h)
                start_y = (bar_y + bar_h//2) - fill_h
                cv2.rectangle(frame, (bar_x + 2, start_y), (bar_x + bar_w - 2, bar_y + bar_h//2), self.COLOR_BLUE, -1)
                
                # Icon (Text)
                cv2.putText(frame, "VOL", (bar_x - 10, bar_y + bar_h//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)