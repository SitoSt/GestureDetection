from typing import Dict, Any, Optional
from .gesture_model import GestureModel


class MockGestureModel(GestureModel):
    def predict(self, payload: Dict[str, Any]) -> Optional[str]:
        return None
