import asyncio
import websockets
import json
from client.actions.music_control import execute_music_action

class WebSocketClient:
    def __init__(self, host="127.0.0.1", port=8765):
        self.uri = f"ws://{host}:{port}"

    async def connect_and_listen(self):
        # Esta función debe correr en un thread o task separado para
        # no bloquear la captura de video en tiempo real.
        try:
            print(f"🔗 Conectando a servidor WebSocket en {self.uri}")
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                print("🟢 Conexión establecida. Escuchando comandos...")
                await self.receive_commands()
        except ConnectionRefusedError:
            print("🔴 Error: Servidor no encontrado o no activo.")
        except Exception as e:
            print(f"❌ Error en la conexión WebSocket: {e}")

    async def send_data(self, data_json: str):
        """
        Envía los landmarks serializados al servidor.
        Se llama desde el bucle de captura de video (video_stream.py).
        """
        if hasattr(self, 'websocket'):
            try:
                await self.websocket.send(data_json)
            except websockets.exceptions.ConnectionClosedOK:
                print("Conexión cerrada. Deteniendo envío.")
            except Exception as e:
                # Es normal que salte un error si el servidor se cae
                pass

    async def receive_commands(self):
        """
        Bucle infinito para escuchar comandos del servidor.
        """
        while True:
            try:
                command_json = await self.websocket.recv()
                command = json.loads(command_json)
                
                # Ejecutar la acción local (pausa, volumen, etc.)
                gesture = command.get("gesture")
                if gesture:
                    execute_music_action(gesture)
                
            except websockets.exceptions.ConnectionClosedOK:
                print("Conexión con el servidor cerrada.")
                break
            except Exception as e:
                print(f"Error al recibir comando: {e}")
                await asyncio.sleep(1) # Esperar antes de reintentar