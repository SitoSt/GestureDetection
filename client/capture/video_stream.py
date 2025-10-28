import cv2
import asyncio
import time
from .landmark_extractor import LandmarkExtractor
from client.ws_client import WebSocketClient
from shared.schemas import serialize_landmarks

class VideoStream:
    def __init__(self, ws_client: WebSocketClient):
        self.extractor = LandmarkExtractor()
        self.ws_client = ws_client
        self.cap = cv2.VideoCapture(0)
        self.FPS = 30 # Tasa de frames objetivo para control (aunque la cámara puede dar más)

    async def start_streaming(self):
        print("🎥 GestureDetection Cliente listo. Iniciando captura de video...")
        
        if not self.cap.isOpened():
            print("🔴 Error: No se puede abrir la cámara.")
            return

        last_time = time.time()
        
        # Usamos asyncio.gather para asegurar que el bucle de captura de video
        # y la conexión/recepción de comandos del WebSocket corran en paralelo
        await asyncio.gather(
            self._video_loop(),
            self.ws_client.connect_and_listen()
        )

    async def _video_loop(self):
        while True:
            # Captura de Frame
            ret, frame = self.cap.read()
            if not ret:
                print("🔴 Error al leer el frame.")
                break

            # Invertir el frame para el efecto espejo (más intuitivo)
            frame = cv2.flip(frame, 1)

            # 1. Extracción de Landmarks
            results = self.extractor.process_frame(frame)

            # 2. Serialización y Envío
            data_json = serialize_landmarks(results)
            
            # Solo enviamos si MediaPipe detectó algo (manos o cuerpo)
            if data_json:
                # Nota: usamos asyncio.ensure_future o una simple llamada await 
                # si no queremos bloquear el frame rate. Aquí usamos await para 
                # priorizar el orden, pero podemos optimizar con ensure_future.
                await self.ws_client.send_data(data_json)

            # 3. Visualización (Opcional)
            self.extractor.draw_landmarks(frame, results)
            cv2.imshow("🖐 GestureDetection Cliente", frame)
            
            # 4. Control de Salida
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Control de Frame Rate (para no sobrecargar la red)
            await asyncio.sleep(1 / self.FPS) 

        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Cliente GestureDetection detenida.")