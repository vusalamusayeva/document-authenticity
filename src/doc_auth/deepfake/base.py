from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BaseDeepfakeDetector(ABC):
    @abstractmethod
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        pass