#type: ignore
import json

def serialize_landmarks(results, frame_b64: str = None):
    """
    Convierte MediaPipe a JSON string plano. SOLO LANDMARKS.
    """
    data = {}

    # --- Manos ---
    if results.multi_hand_landmarks:
        # Solo tomamos la primera mano
        lm_list = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            lm_list.extend([landmark.x, landmark.y, landmark.z])
        data['hands'] = lm_list

    # --- Cuerpo (Pose) ---
    if results.pose_landmarks:
        # Añadido para escalabilidad
        data['pose'] = [val for lm in results.pose_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
    
    # --- Frame para futura escalabilidad ---
    if frame_b64:
        data['frame'] = frame_b64 
        
    if not data:
        return None

    return json.dumps(data)

# Comando que el Servidor devuelve al Cliente (sin cambios)
def create_command_json(gesture: str):
    """Crea el comando JSON que el servidor envía al cliente."""
    return json.dumps({"gesture": gesture})