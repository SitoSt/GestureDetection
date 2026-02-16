from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class GestureModel(ABC):
    @abstractmethod
    def predict(self, payload: Dict[str, Any]) -> Optional[str]:
        """Return gesture label or None when no match."""
        raise NotImplementedError
