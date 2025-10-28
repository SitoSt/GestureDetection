#type: ignore
import mediapipe as mp
import cv2

class LandmarkExtractor:
    def __init__(self):
        # Inicializar el modelo de manos y cuerpo de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # Manos: solo necesitamos una mano para la detección de gestos básica
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Pose (Cuerpo): Para futura escalabilidad
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """
        Convierte el frame a RGB y extrae los landmarks de manos y cuerpo.
        Devuelve el objeto 'results' de MediaPipe.
        """
        # MediaPipe espera RGB, OpenCV usa BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesamiento en paralelo (o secuencial rápido)
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)

        # Combina los resultados en un solo objeto para simplificar el envío
        class CombinedResults:
            def __init__(self, hand_res, pose_res):
                self.multi_hand_landmarks = hand_res.multi_hand_landmarks
                self.pose_landmarks = pose_res.pose_landmarks
        
        return CombinedResults(hand_results, pose_results)

    def draw_landmarks(self, frame, results):
        """
        Dibuja los landmarks de manos y cuerpo en el frame para visualización.
        """
        mp_draw = mp.solutions.drawing_utils
        
        # Dibujar Manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Dibujar Pose (Cuerpo)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)