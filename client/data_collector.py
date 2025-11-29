import cv2
import json
import time
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from client.capture.landmark_extractor import LandmarkExtractor
from shared.schemas import serialize_landmarks

DATA_DIR = "training_data"
SEQUENCE_DURATION = 3 # Duración de la secuencia a grabar (segundos)
FPS_TARGET = 15      # Frames por segundo de muestreo (15-20 es ideal para secuencias)

# Clases que la IA debe aprender a clasificar
GESTURE_CLASSES = [
    "play_pause_INTENCIONAL",       # Puño cerrado limpio.
    "next_track_INTENCIONAL",       # Pulgar apuntando a la derecha.
    "prev_track_INTENCIONAL",       # Pulgar apuntando a la izquierda.
    "open_spotify_INTENTIONAL",     # Gesto de Índice y Meñique limpio
    "falso_positivo_FUMAR",         # Puño cerca de la boca (para anular Play/Pause).
    "falso_positivo_RASCAR",        # Puño cerca de la cara/cabeza.
    "no_accion_MANO_ABIERTA",       # Mano abierta en reposo (para anular Next Track).
    "no_accion_MOVIMIENTO_ALEATORIO"# Mover la mano sin intención de gesto.
]

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    extractor = LandmarkExtractor()
    # Usamos cap_api=cv2.CAP_DSHOW en Windows o cv2.CAP_ANY por defecto.
    # En macOS, cv2.VideoCapture(0) suele ser suficiente.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[DataCollector] Error: Cannot open camera.")
        return

    print("[DataCollector] Training Data Collector Started")
    print("=" * 50)

    while True:
        print("\nSelect a gesture class to record:")
        for i, name in enumerate(GESTURE_CLASSES):
            print(f"  {i}: {name}")
        print("  q: Quit")
        
        choice = input("\nEnter class number or 'q': ").strip()
        
        if choice.lower() == 'q':
            break

        try:
            class_index = int(choice)
            if 0 <= class_index < len(GESTURE_CLASSES):
                class_name = GESTURE_CLASSES[class_index]
                collect_sequence(cap, extractor, class_name)
            else:
                print("[DataCollector] Invalid selection.")
        except ValueError:
            print("[DataCollector] Invalid input.")
        
    cap.release()
    cv2.destroyAllWindows()
    print("\n[DataCollector] Shutdown.")


def collect_sequence(cap, extractor, class_name: str):
    """Collect a timed sequence of gesture data."""
    print(f"\n[DataCollector] Preparing to record: '{class_name}'")
    print("Position your hand and hold the gesture. Recording starts in 3 seconds...")
    time.sleep(3)
    
    start_time = time.time()
    sequence_data = []
    frame_delay = 1 / FPS_TARGET
    last_frame_time = 0

    print(f"[DataCollector] Recording for {SEQUENCE_DURATION}s...")

    while time.time() < start_time + SEQUENCE_DURATION:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        current_time = time.time()
        if current_time - last_frame_time < frame_delay:
            extractor.draw_landmarks(frame, extractor.process_frame(frame))
            cv2.imshow("Data Collector", frame)
            cv2.waitKey(1)
            continue
            
        last_frame_time = current_time

        results = extractor.process_frame(frame)
        frame_data = {'raw_gesture': class_name}
        
        if results.multi_hand_landmarks:
            lm_list = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                lm_list.extend([landmark.x, landmark.y, landmark.z])
            frame_data['hands'] = lm_list
        
        if results.pose_landmarks:
            pose_list = []
            for landmark in results.pose_landmarks.landmark:
                pose_list.extend([landmark.x, landmark.y, landmark.z])
            frame_data['pose'] = pose_list
            
        if frame_data.get('hands') or frame_data.get('pose'):
            sequence_data.append(frame_data)

        extractor.draw_landmarks(frame, results)
        cv2.putText(frame, f"Recording: {class_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Data Collector", frame)
        cv2.waitKey(1)

    print("[DataCollector] Recording complete.")

    timestamp = int(time.time())
    class_safe_name = class_name.replace(' ', '_').replace('/', '_')
    file_path = os.path.join(DATA_DIR, f"{class_safe_name}_{timestamp}.json")
    
    final_data = {
        "class": class_name,
        "duration_s": SEQUENCE_DURATION,
        "fps": FPS_TARGET,
        "sequence": sequence_data
    }
    
    with open(file_path, 'w') as f:
        json.dump(final_data, f, indent=4)
    
    print(f"[DataCollector] Saved: {file_path}")
    print(f"[DataCollector] Frames captured: {len(sequence_data)}")


if __name__ == "__main__":
    main()