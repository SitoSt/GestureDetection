import asyncio
from client.ws_client import WebSocketClient
from client.capture.video_stream import VideoStream

SERVER_HOST = "green-house.local"
SERVER_PORT = 8765

def main():
    ws_client = WebSocketClient(host=SERVER_HOST, port=SERVER_PORT)
    video_stream = VideoStream(ws_client)
    
    try:
        asyncio.run(video_stream.start_streaming())
    except KeyboardInterrupt:
        print("\n[Client] Shutdown by user.")
    except Exception as e:
        print(f"[Client] Fatal error: {e}")

if __name__ == "__main__":
    main()