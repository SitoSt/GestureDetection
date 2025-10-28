# server/modules/gestures.py (El Cerebro de la Detección - V3 con 3D)
import numpy as np
import time
from typing import Tuple, Optional

class GestureProcessor:
    def __init__(self):
        # --- Configuración de Filtrado ---
        self.PINCH_THRESHOLD_3D = 0.045  # Distancia 3D más estricta
        self.COOLDOWN = 0.8  
        self.GESTURE_STABILITY_FRAMES = 5
        self.VOLUME_MOVE_THRESHOLD = 0.025 # Umbral de movimiento vertical

        # --- Estado Interno ---
        self.last_action_time = 0
        self.last_index_y: Optional[float] = None # Para control de volumen
        
        # --- Lógica de Estabilidad ---
        self.current_stable_gesture: Optional[str] = None
        self.gesture_count = 0

    def _get_coords(self, lm_list: list, index: int) -> Tuple[float, float, float]:
        """Extrae las coordenadas x, y, z de un landmark específico (index)."""
        # Las coordenadas están serializadas como [x0, y0, z0, x1, y1, z1, ...]
        if len(lm_list) < (index * 3 + 2):
            # Devolvemos un valor fuera de rango si la lista está incompleta.
            return 2.0, 2.0, 0.0 
        x = lm_list[index * 3]
        y = lm_list[index * 3 + 1]
        z = lm_list[index * 3 + 2]
        return x, y, z

    def _get_distance_3d(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calcula la distancia euclídea 3D entre dos puntos."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


    def _detect_raw_gesture(self, lm_list: list) -> Optional[str]:
        """
        Detecta el gesto basado en la forma 3D y 2D (Y) de la mano.
        """
        # 1. Extracción de Puntos Clave
        try:
            # Puntas (3D)
            thumb_tip = self._get_coords(lm_list, 4)
            index_tip = self._get_coords(lm_list, 8)
            middle_tip = self._get_coords(lm_list, 12)
            ring_tip = self._get_coords(lm_list, 16)
            
            # Articulaciones PIP (Para chequeo de doblado/extendido - solo Y)
            index_pip_y = self._get_coords(lm_list, 6)[1]
            middle_pip_y = self._get_coords(lm_list, 10)[1]
            ring_pip_y = self._get_coords(lm_list, 14)[1]
            pinky_pip_y = self._get_coords(lm_list, 18)[1]

            wrist_y = self._get_coords(lm_list, 0)[1]
        except IndexError:
            return None
        
        # Distancia 3D Pulgar-Índice
        dist_thumb_index_3d = self._get_distance_3d(thumb_tip, index_tip)
        
        # --- Detecciones (Mutuamente Excluyentes) ---

        # 1. Detección de PINCH (Activación del Volumen)
        # Condición: Pulgar e Índice muy cerca en el espacio 3D.
        if dist_thumb_index_3d < self.PINCH_THRESHOLD_3D:
            return "pinch_active"

        # 2. Detección de NEXT TRACK (DOS DEDOS)
        # Condición: Índice y Corazón extendidos, Anular y Meñique cerrados.
        index_extended = index_tip[1] < index_pip_y
        middle_extended = middle_tip[1] < middle_pip_y
        ring_closed = ring_tip[1] > ring_pip_y
        pinky_closed = ring_tip[1] > pinky_pip_y # Meñique suele estar cerrado con el anular

        if index_extended and middle_extended and ring_closed and pinky_closed:
             return "next_track_shape"


        # 3. Detección de PLAY/PAUSE (PUÑO CERRADO)
        # Condición 1 (Cerrado): Todos los dedos principales están más abajo (mayor Y) que sus articulaciones.
        index_closed = index_tip[1] > index_pip_y
        middle_closed = middle_tip[1] > middle_pip_y
        ring_closed_check = ring_tip[1] > ring_pip_y
        
        # Condición 2 (No Pinza): Pulgar no está cerca del índice.
        is_not_pinch = dist_thumb_index_3d > 0.1
        
        # Condición 3 (Profundidad - Robustez del Puño): El nudillo del índice (6) es más profundo (menor Z) que la punta del pulgar (4).
        # Esto ayuda a distinguir la mano plana de un puño.
        index_knuckle_z = self._get_coords(lm_list, 6)[2]
        thumb_tip_z = thumb_tip[2]
        
        thumb_is_tucked = index_knuckle_z < thumb_tip_z # Pulgar "dentro" del puño

        if index_closed and middle_closed and ring_closed_check and is_not_pinch and thumb_is_tucked:
            return "play_pause_shape"

        return None


    def _handle_volume(self, index_tip_y: float) -> Optional[str]:
        """
        Gestiona el control de volumen basado en el movimiento vertical (Y) del índice.
        """
        y = index_tip_y

        if self.last_index_y is None:
            self.last_index_y = y
            return None

        # dy es la diferencia de movimiento: Negativo = Moviendo Hacia Abajo (volumen down)
        dy = self.last_index_y - y 
        
        if dy > self.VOLUME_MOVE_THRESHOLD: # Subiendo (Y disminuye)
            self.last_index_y = y 
            return "volume_up"
        elif dy < -self.VOLUME_MOVE_THRESHOLD: # Bajando (Y aumenta)
            self.last_index_y = y
            return "volume_down"
            
        return None

    def is_gesture_contextually_valid(self, data: dict) -> bool:
        """
        Lógica contextual (pre-IA) para anular un gesto (Falsos Positivos: Fumar/Rascarse).
        Solo se ejecuta para Play/Pause. Requiere landmarks de Pose (Cuerpo).
        """
        if self.current_stable_gesture != "play_pause_shape":
             return True 
             
        if 'pose' not in data or 'hands' not in data:
            return True

        pose_lm = data['pose']
        hand_lm = data['hands']
            
        try:
            # Puntos de la cabeza (Pose Landmarks)
            # El esquema de pose es [x0, y0, z0, x1, y1, z1, ...]
            nose_y = pose_lm[1]
            nose_x = pose_lm[0]
            
            # Punto de la Muñeca (Hand Landmark 0)
            wrist_x, wrist_y, _ = self._get_coords(hand_lm, 0)
            
            # Condición de Falso Positivo: La muñeca está en proximidad del rostro.
            # Umbral de 0.25 en X y Y para la proximidad del rostro.
            if abs(wrist_y - nose_y) < 0.25 and abs(wrist_x - nose_x) < 0.25:
                # El puño cerca del rostro NO es un comando.
                return False
                    
        except IndexError:
            return True
            
        return True
    
    
    def process_landmarks(self, data: dict) -> Optional[str]:
        """
        Función principal de procesamiento: aplica filtrado, estabilidad y contexto.
        """
        now = time.time()
        
        if now - self.last_action_time < self.COOLDOWN:
            return None

        if 'hands' not in data:
            self.current_stable_gesture = None
            self.gesture_count = 0
            self.last_index_y = None
            return None

        lm_list = data['hands']
        index_tip_y = self._get_coords(lm_list, 8)[1] # Solo necesitamos Y para volumen
        
        # --- 1. Detección de Forma Cruda ---
        raw_gesture = self._detect_raw_gesture(lm_list)
        
        # --- 2. Manejo del PINCH (Volumen) ---
        if raw_gesture == "pinch_active":
            # Si hay pinza, anular el contador de estabilidad de otros gestos.
            self.current_stable_gesture = None
            self.gesture_count = 0
            
            volume_command = self._handle_volume(index_tip_y)
            if volume_command:
                self.last_action_time = now
                return volume_command
            return None
        else:
            # Si no hay pinch, resetea la base del volumen
            self.last_index_y = None

        # --- 3. Lógica de Estabilidad (Play/Pause y Next Track) ---
        
        is_shape_gesture = raw_gesture in ["play_pause_shape", "next_track_shape"]

        if is_shape_gesture and raw_gesture == self.current_stable_gesture:
             self.gesture_count += 1
        elif is_shape_gesture and raw_gesture != self.current_stable_gesture:
             self.current_stable_gesture = raw_gesture
             self.gesture_count = 1
             return None
        elif raw_gesture is None:
            self.current_stable_gesture = None
            self.gesture_count = 0
            
        # 4. Confirmación por Estabilidad y Contexto
        if self.gesture_count >= self.GESTURE_STABILITY_FRAMES:
             
             if self.is_gesture_contextually_valid(data):
                # Gesto confirmado y validado
                command = self.current_stable_gesture.replace("_shape", "")
                self.last_action_time = now
                self.gesture_count = 0 
                self.current_stable_gesture = None
                
                print(f"✅ GESTO CONFIRMADO Y VALIDADO: {command}")
                return command
             else:
                 # Gesto estable, pero la lógica contextual lo anula (ej. mano cerca de la cara)
                 print("❌ FALSO POSITIVO DESCARTADO POR LÓGICA CONTEXTUAL.")
                 self.gesture_count = 0
                 self.current_stable_gesture = None
                 return None
                 
        return None