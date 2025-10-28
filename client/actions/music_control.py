# En un sistema real, aquí usaríamos el módulo 'keyboard' o comandos
# específicos de macOS para el control de música/volumen.
def execute_music_action(gesture: str):
    """
    Ejecuta una acción local basada en el comando del servidor.
    """
    action_map = {
        "play_pause": "⏯️ Toggling Play/Pause...",
        "next_track": "⏭️ Moving to Next Track...",
        "prev_track": "⏮️ Moving to Previous Track...",
        "volume_up": "🔊 Increasing Volume (Placeholder)...",
        "volume_down": "🔉 Decreasing Volume (Placeholder)..."
    }

    if gesture in action_map:
        print(f"✅ GESTO DETECTADO: {action_map[gesture]} [Servidor: {gesture}]")
    else:
        print(f"⚠️ Comando desconocido: {gesture}")