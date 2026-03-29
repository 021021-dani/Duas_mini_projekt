"""
Module: classify_tile.py
Responsibility: High-level terrain classification using ColorExtractor.
"""

import numpy as np

# from color.detec import ColorExtractor


class TileClassifier:
    """Classifies the terrain of a single cell."""

    def __init__(self, color_extractor):
        self.color_extractor = color_extractor

    def classify(self, cell_image: np.ndarray) -> str:
        """
        TODO:
        1. Get HSV.
        2. Evaluate mask coverage for all terrains.
        3. EDGE CASE: Fallback to texture/SIFT if max coverage is too low (uncertain).
        Returns string terrain name.
        """
        pass
