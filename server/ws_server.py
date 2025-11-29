import asyncio
import websockets
import json
from shared.schemas import create_command_json
from .modules.gestures import GestureProcessor

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765

processor = GestureProcessor()

async def handle_client(websocket, path):
    """Handle incoming WebSocket connections and process gesture data."""
    print("[Server] Client connected.")
    try:
        async for message in websocket:
            data = json.loads(message)
            
            gesture = processor.process_landmarks(data)
            
            if gesture:
                command_json = create_command_json(gesture)
                await websocket.send(command_json)
                print(f"[Server] Sent gesture: {gesture}")

    except websockets.exceptions.ConnectionClosed:
        print("[Server] Client disconnected.")
    except Exception as e:
        print(f"[Server] Error in handler: {e}")

async def start_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """Start the WebSocket server for gesture detection."""
    server = websockets.serve(handle_client, host, port)
    print(f"[Server] GestureDetection server running on ws://{host}:{port}")
    await server
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(start_server())