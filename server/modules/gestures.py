#type: ignore
import numpy as np
from typing import Tuple, Optional

class GestureProcessor:
    def __init__(self):
        # Umbrales y estado
        self.PINCH_THRESHOLD = 0.05
        self.last_index_y: Optional[float] = None
        
        # Lógica para evitar Falsos Positivos: Añadimos un buffer de secuencia
        self.last_actions = []
        self.action_history_len = 5 # Para evitar repeticiones rápidas

    def _get_coords(self, lm_list: list, index: int) -> Tuple[float, float]:
        """Extrae las coordenadas x, y de un landmark específico (index)."""
        if len(lm_list) < (index * 3 + 2):
            raise IndexError("Lista de landmarks incompleta.")
        x = lm_list[index * 3]
        y = lm_list[index * 3 + 1]
        return x, y

    def process_landmarks(self, data: dict) -> str or None:
        """
        Lógica de Detección Centralizada: Simple, secuencial y contextual (a futuro).
        """
        
        # 1. Preparación de datos (Landmarks de Manos)
        if 'hands' not in data:
            # Aquí la IA podría detectar postura corporal sin manos
            return None
        
        lm_list = data['hands'] # Lista plana de 63 floats (21 * 3)
        
        try:
            thumb_tip = self._get_coords(lm_list, 4)
            index_tip = self._get_coords(lm_list, 8)
            middle_tip = self._get_coords(lm_list, 12)
            wrist = self._get_coords(lm_list, 0)
        except IndexError:
            return None

        # Distancia euclídea normalizada entre pulgar e índice (solo en 2D)
        dist_thumb_index = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        
        detected_gesture = None
        
        # 2. Detección de Pinch y Volumen (Ahora en el Servidor)
        if dist_thumb_index < self.PINCH_THRESHOLD:
            # Lógica de volumen: requiere secuencia de movimiento para ser detectado
            if self.last_index_y is not None:
                dy = self.last_index_y - index_tip[1] 
                
                if dy > 0.015: # Subiendo (más umbral para reducir jitter)
                    self.last_index_y = index_tip[1]
                    detected_gesture = "volume_up"
                elif dy < -0.015: # Bajando
                    self.last_index_y = index_tip[1]
                    detected_gesture = "volume_down"
            else:
                self.last_index_y = index_tip[1]
        else:
            self.last_index_y = None

        # 3. Detección de "Dos Dedos" (Siguiente Canción)
        if not detected_gesture and index_tip[1] < thumb_tip[1] and middle_tip[1] < thumb_tip[1] and index_tip[1] < wrist[1]:
            detected_gesture = "next_track"
            
        # 4. Detección de "Puño" (Play/Pause)
        # Esto es lo que genera falsos positivos, la IA lo filtrará después.
        if not detected_gesture and index_tip[1] > wrist[1] and dist_thumb_index > 0.1:
             detected_gesture = "play_pause"


        # 5. LÓGICA CONTEXTUAL Y DE SECUENCIA (FUTURO DE IA)
        # Aquí se integrará el modelo de IA o la lógica secuencial para:
        # a) Filtrar si el puño está cerca de la boca (fumando).
        # b) Comprobar si la mano ha estado en una posición estable antes del gesto.
        # c) Asegurar que el gesto se mantiene en 'X' frames.
        
        # Placeholder de filtrado simple: Evitar repetición instantánea del mismo comando
        if detected_gesture:
             if self.last_actions and self.last_actions[-1] == detected_gesture:
                 self.last_actions.append(detected_gesture)
                 return None # Ignorar si es el mismo gesto inmediatamente repetido
             
             self.last_actions.append(detected_gesture)
             self.last_actions = self.last_actions[-self.action_history_len:] # Mantiene el buffer limpio
             
             return detected_gesture

        return None