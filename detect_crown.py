"""
Modul: Kronedetektion ved hjælp af farvemaskering og konturgenkendelse.
"""

import cv2
import numpy as np


class CrownDetector:
    """Detekterer og tæller antallet af kroner på en enkelt tile."""

    def __init__(self, min_area=50, max_area=800):
        # Definerer farveområdet for den gyldne krone i HSV-format
        # (Disse værdier kan trimmes afhængigt af belysningen i datasættet)
        self.lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
        self.upper_yellow = np.array([35, 255, 255], dtype=np.uint8)

        # Arealgrænser for at filtrere støj og undgå falske positiver
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, cell_image: np.ndarray) -> int:
        """
        Tæller antallet af gyldne kroner på et tile-billede.
        Returnerer et heltal der repræsenterer antallet (typisk 0-3).
        """

        # Hvis billedet er tomt, returneres 0
        if cell_image is None or cell_image.size == 0:
            return 0

        # Konverter billedet fra BGR (standard i OpenCV) til HSV for bedre farveseparation
        hsv_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV)

        # Opret en binær maske, der kun isolerer de gule/gyldne farver
        mask = cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow)

        # Rens masken med morfologiske operationer (fjerner støj)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find konturer (sammenhængende gule områder) i den rensede maske
        contours, _ = cv2.findContours(
            mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        crown_count = 0

        # Gennemgå alle fundne konturer for at tælle de gyldige kroner
        for contour in contours:
            area = cv2.contourArea(contour)

            # Tæl kun konturen som en krone, hvis arealet er inden for de forventede grænser
            if self.min_area <= area <= self.max_area:
                crown_count += 1

        # I Kingdomino er der maksimalt 3 kroner på en brik
        return min(crown_count, 3)
