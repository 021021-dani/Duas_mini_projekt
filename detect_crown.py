"""
Module: detect_crown.py
Responsibility: Extract crown multiplier counts from a cell.
"""

import numpy as np


class CrownDetector:
    """Detects and counts crowns within a cell."""

    def detect(self, cell_image: np.ndarray, terrain: str) -> int:
        """
        TODO:
        1. Ignore if terrain is Castle/Empty.
        2. Filter for gold/yellow crown shapes using contour area constraints.
        3. EDGE CASE: Cap value at 3. Return int between 0-3.
        """
        pass
