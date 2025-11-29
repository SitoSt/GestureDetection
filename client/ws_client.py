import asyncio
import websockets
import json
from typing import Optional
from client.actions.action_executor import execute_action

class WebSocketClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.uri = f"ws://{host}:{port}"
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None

    async def connect_and_listen(self) -> None:
        """Establish WebSocket connection and listen for commands from server."""
        try:
            print(f"[WebSocket] Connecting to {self.uri}")
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                print("[WebSocket] Connected. Listening for commands...")
                await self._receive_commands()
        except ConnectionRefusedError:
            print("[WebSocket] Error: Server not found or not active.")
        except Exception as e:
            print(f"[WebSocket] Connection error: {e}")

    async def send_data(self, data_json: str) -> None:
        """Send serialized landmarks to the server."""
        if self.websocket:
            try:
                await self.websocket.send(data_json)
            except websockets.exceptions.ConnectionClosedOK:
                print("[WebSocket] Connection closed. Stopping data transmission.")
            except Exception:
                pass

    async def _receive_commands(self) -> None:
        """Listen for and execute commands received from the server."""
        while True:
            try:
                command_json = await self.websocket.recv()
                command = json.loads(command_json)
                
                gesture = command.get("gesture")
                if gesture:
                    execute_action(gesture)
                
            except websockets.exceptions.ConnectionClosedOK:
                print("[WebSocket] Server connection closed.")
                break
            except Exception as e:
                print(f"[WebSocket] Error receiving command: {e}")
                await asyncio.sleep(1)