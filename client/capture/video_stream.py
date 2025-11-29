import cv2
import asyncio
from typing import Optional
from .landmark_extractor import LandmarkExtractor
from client.ws_client import WebSocketClient
from shared.schemas import serialize_landmarks

class VideoStream:
    TARGET_FPS = 30
    WINDOW_NAME = "GestureDetection Client"
    
    def __init__(self, ws_client: WebSocketClient):
        self.extractor = LandmarkExtractor()
        self.ws_client = ws_client
        self.cap = cv2.VideoCapture(0)

    async def start_streaming(self) -> None:
        """Start video capture and WebSocket communication in parallel."""
        print("[VideoStream] Starting video capture...")
        
        if not self.cap.isOpened():
            print("[VideoStream] Error: Cannot open camera.")
            return
        
        await asyncio.gather(
            self._capture_loop(),
            self.ws_client.connect_and_listen()
        )

    async def _capture_loop(self) -> None:
        """Main loop for capturing frames, processing landmarks, and sending data."""
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("[VideoStream] Error reading frame.")
                break

            frame = cv2.flip(frame, 1)

            results = self.extractor.process_frame(frame)
            data_json = serialize_landmarks(results)
            
            if data_json:
                # Nota: usamos asyncio.ensure_future o una simple llamada await 
                # si no queremos bloquear el frame rate. Aquí usamos await para 
                # priorizar el orden, pero podemos optimizar con ensure_future.
                await self.ws_client.send_data(data_json)

            self.extractor.draw_landmarks(frame, results)
            cv2.imshow(self.WINDOW_NAME, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(1 / self.TARGET_FPS)

        self.cap.release()
        cv2.destroyAllWindows()
        print("[VideoStream] Video capture stopped.")