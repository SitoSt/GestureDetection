# server/ws_server.py
import asyncio
import websockets
import json
from shared.schemas import create_command_json
from .modules.gestures import GestureProcessor

# Instancia del procesador de gestos
processor = GestureProcessor()

async def handler(websocket, path):
    print("🟢 Cliente conectado.")
    try:
        async for message in websocket:
            # 1. Recibir Landmarks
            data = json.loads(message)
            
            # 2. Procesar el Gesto (Lógica Simple Inicial)
            # Pasamos los datos al módulo de detección de gestos
            gesture = processor.process_landmarks(data)
            
            # 3. Devolver Comando al Cliente
            if gesture:
                command_json = create_command_json(gesture)
                await websocket.send(command_json)
                print(f"SENT: {gesture}")

    except websockets.exceptions.ConnectionClosed:
        print("🔴 Cliente desconectado.")
    except Exception as e:
        print(f"❌ Error en el handler del servidor: {e}")

async def start_server(host="0.0.0.0", port=8765):
    # Host 0.0.0.0 permite escuchar en todas las interfaces de red (Ethernet, Wi-Fi)
    start_server = websockets.serve(handler, host, port)
    print(f"💻 Servidor GestureDetection activo en ws://{host}:{port}")
    await start_server
    await asyncio.Future() # Mantener el servidor corriendo

if __name__ == "__main__":
    # Comando para ejecutar el servidor: python server/ws_server.py
    asyncio.run(start_server())