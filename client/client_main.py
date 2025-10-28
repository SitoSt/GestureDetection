import asyncio
from client.ws_client import WebSocketClient
from client.capture.video_stream import VideoStream

# Configuración de la dirección del servidor (tu Ubuntu Server)
SERVER_HOST = "green-house" # Cambia esto a la IP local de tu Ubuntu Server (ej. "192.168.1.10")
SERVER_PORT = 8765

def main():
    # 1. Inicializar el cliente WebSocket
    ws_client = WebSocketClient(host=SERVER_HOST, port=SERVER_PORT)
    
    # 2. Inicializar el Stream de Video
    video_stream = VideoStream(ws_client)
    
    # 3. Arrancar la conexión y el stream en el mismo loop de asyncio
    try:
        asyncio.run(video_stream.start_streaming())
    except KeyboardInterrupt:
        print("\nCliente apagado por el usuario.")
    except Exception as e:
        print(f"Error fatal en el cliente: {e}")

if __name__ == "__main__":
    main()