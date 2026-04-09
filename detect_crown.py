"""
Modul: Kronedetektion ved hjælp af Template Matching.

Dette modul er designet til at identificere og tælle kongekroner på KingDomino-tiles.
"""

from pathlib import Path

import cv2
import numpy as np

# Definer de spilleplader, der sorteres fra til test-sættet
TEST_BOARDS = {
    "board_1",
    "board_5",
    "board_19",
    "board_23",
    "board_25",
    "board_29",
    "board_35",
    "board_39",
    "board_49",
    "board_53",
    "board_67",
    "board_70",
}


class CrownDetector:
    """
    Finder og tæller antallet af kroner på en tile, 
    ved at trække templaten over pixlerne med sliding window.
    """

    def __init__(self, template_path="high_res_crown.png", threshold=0.88):
        """
        Args:
            template_path (str): Den absolutte eller relative sti til .png skabelonen.
            threshold (float): Tærskelværdi for accepterede overlap (mellem 0.0 og 1.0).
                               Højere minimerer falsk positiver, lavere fanger flere.
        """
        self.threshold = threshold
        self.template_path = template_path

        # 1. Indlæs skabelonen (cv2.IMREAD_UNCHANGED beholder alfa-kanalen / gennemsigtighed)
        template_img = cv2.imread(self.template_path, cv2.IMREAD_UNCHANGED)

        if template_img is None:
            raise FileNotFoundError(
                f"Kunne ikke indlæse skabelonen: {self.template_path}"
            )

        self.template_gray = None
        self.mask = None

        # 2. Håndter PNG-billeder der indeholder gennemsigtighed (shape=[H, W, 4 kanaler])
        if len(template_img.shape) == 3 and template_img.shape[2] == 4:
            # Opdel i BGR (grafik) og Alfa (Gennemsigtighed)
            bgr = template_img[:, :, :3]
            alpha = template_img[:, :, 3]

            # Konverter RGB grafikken til Gråtoneskal for "Template Matching"
            self.template_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Alfa-kanalen fungerer som en pixel-maske
            self.mask = alpha
        else:
            # Standard konvertering
            self.template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        # Gem template dimensioner til non-max-suppression funktionen
        self.template_h, self.template_w = self.template_gray.shape[:2]

    def detect(self, cell_image: np.ndarray) -> int:
        """
        Tæller antallet af kroner vha. Normalized Cross Correlation.
        Returnerer et heltal mellem 0 og 3.
        """
        if cell_image is None or cell_image.size == 0:
            return 0

        # Processér inputbrikken til Gråtone
        gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

        # 3. Anvend Template Matching
        # Hvis mask er tilgængelig, bruger vi cv2.TM_CCORR_NORMED
        if self.mask is not None:
            res = cv2.matchTemplate(
                gray_cell, self.template_gray, cv2.TM_CCORR_NORMED, mask=self.mask
            )
        else:
            res = cv2.matchTemplate(gray_cell, self.template_gray, cv2.TM_CCOEFF_NORMED)

        # 4. Find de Array-koordinater som overskrider threshold værdien
        loc = np.where(res >= self.threshold)
        points = list(zip(*loc[::-1]))

        if not points:
            return 0

        # 5. Non-Maximum Suppression (Overlapping Bounding Boxes)
        rectangles = []
        for x, y in points:
            rectangles.append(
                [int(x), int(y), int(self.template_w), int(self.template_h)]
            )
            rectangles.append(
                [int(x), int(y), int(self.template_w), int(self.template_h)]
            )

        rectangles, weights = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.2)

        crown_count = len(rectangles)

        # Kingdomino har maksimalt 3 kroner per brik
        return min(crown_count, 3)


if __name__ == "__main__":
    # --- Akademisk Test af CrownDetector på Træning/Validering ---
    print("Initialiserer Template Matching (vha. high_res_crown.png)...")
    try:
        detector = CrownDetector(template_path="high_res_crown.png", threshold=0.88)
    except FileNotFoundError as e:
        print(f"Fejl: {e}")
        exit(1)

    tiles_dir = Path("KD_tiles")
    if not tiles_dir.exists():
        print(f"\nMappe med tiles blev ikke fundet ved: {tiles_dir}")
        exit(1)

    processed_tiles = 0
    total_crowns = 0

    print("\nAnalyserer spilleplader (uden TEST_BOARDS)...")

    for board_folder in sorted(tiles_dir.iterdir()):
        if not board_folder.is_dir():
            continue

        board_name = board_folder.name

        # Forhinderer data leakage
        if board_name in TEST_BOARDS:
            continue

        for tile_file in sorted(board_folder.glob("*.jpg")):
            tile_img = cv2.imread(str(tile_file))

            if tile_img is None:
                continue

            crowns_found = detector.detect(tile_img)

            processed_tiles += 1
            total_crowns += crowns_found

    print(f"  - Antal Brikker (Tiles) Scannet: {processed_tiles}")
    print(f"  - Estimeret Kongekroner Samlet : {total_crowns}")

    # Kan være at vi skal prøve at finjustere threshold, 
    # eller alternativt prøve med hybridløsninger + farvemaskering
