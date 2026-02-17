import asyncio
import websockets
import json
from typing import Optional
from client.actions.action_executor import execute_action

class WebSocketClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, client_id: str = "client", on_command_callback=None):
        self.uri = f"ws://{host}:{port}"
        self.client_id = client_id
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.on_command_callback = on_command_callback
        self.message_queue = asyncio.Queue(maxsize=30) # Buffer approx 1 sec at 30fps

    async def connect_and_listen(self) -> None:
        """Establish WebSocket connection and manage send/receive loops."""
        try:
            print(f"[WebSocket] Connecting to {self.uri}")
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                print("[WebSocket] Connected. Ready.")
                
                # Run sender and receiver concurrently
                await asyncio.gather(
                    self._receive_commands(),
                    self._sender_loop()
                )
        except ConnectionRefusedError:
            print("[WebSocket] Error: Server not found or not active.")
        except Exception as e:
            print(f"[WebSocket] Connection error: {e}")

    async def send_data(self, data_json: str) -> None:
        """Queue serialized landmarks for sending (Non-blocking)."""
        if self.websocket:
            try:
                # If queue is full, drop the oldest frame (or just fail to put new one)
                # Dropping new frame is better for real-time (keep latency low)
                self.message_queue.put_nowait(data_json)
            except asyncio.QueueFull:
                # print("[WebSocket] Drop frame (Queue Full)")
                pass
            except Exception:
                pass

    async def _sender_loop(self) -> None:
        """Consume queue and send messages."""
        while self.websocket and self.websocket.open:
            try:
                data_json = await self.message_queue.get()
                await self.websocket.send(data_json)
                self.message_queue.task_done()
            except websockets.exceptions.ConnectionClosedOK:
                break
            except Exception as e:
                print(f"[WebSocket] Send error: {e}")
                break

    async def _receive_commands(self) -> None:
        """Listen for and execute commands received from the server."""
        while self.websocket and self.websocket.open:
            try:
                command_json = await self.websocket.recv()
                command = json.loads(command_json)
                
                gesture = command.get("gesture")
                if gesture:
                    execute_action(gesture)
                    if self.on_command_callback:
                        self.on_command_callback(gesture)
                
            except websockets.exceptions.ConnectionClosedOK:
                print("[WebSocket] Server connection closed.")
                break
            except Exception as e:
                print(f"[WebSocket] Error receiving command: {e}")
                # Don't sleep if connection is closed
                if not self.websocket or not self.websocket.open:
                    break
                await asyncio.sleep(1)