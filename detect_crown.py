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

    def __init__(self, template_path="high_res_crown.png", threshold=0.95):
        """
        Args:
            template_path (str): Den absolutte eller relative sti til .png skabelonen.
            threshold (float): Tærskelværdi for accepterede overlap (mellem 0.0 og 1.0).
                               Højere minimerer falske positiver, lavere fanger flere.
        """
        self.threshold = threshold
        self.template_path = template_path

        # 1. Indlæs skabelonen (cv2.IMREAD_UNCHANGED beholder alfa-kanalen / gennemsigtighed)
        template_img = cv2.imread(self.template_path, cv2.IMREAD_UNCHANGED)

        if template_img is None:
            raise FileNotFoundError(
                f"Skabelon er ikke fundet. Path: {self.template_path}"
            )

            # Variabel til gråtone-version af skabelonen. Den sættes til None indtil billedet er behandlet.
        self.template_gray = None

        # Variabel til gennemsigtighed (alfa-maske).
        self.mask = None

        # 2. Håndter PNG-billeder der indeholder gennemsigtighed (shape=[højde, bredde, 4 kanaler(BGR+alfa)])
        # len(template_img.shape) == 3 and template_img.shape[2] == 4
        # Opdel i BGR (grafik) og alfa (Gennemsigtighed)
        bgr = template_img[:, :, :3]
        alpha = template_img[:, :, 3]

        # Konverter BGR til gråtone
        self.template_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Alfa-kanalen fungerer som en pixel-maske
        # Programmet skal ignorere den usynlige baggrund omkring vores krone-skabelon.
        self.mask = alpha

        # Gem template dimensioner til non-max-suppression funktionen
        # Vi udtrækker højde (h) og bredde (b) fra skabelonens dimensioner (shape).
        # Så vi senere kan tegne præcise firkanter (bounding boxes) omkring de kroner vi finder.
        self.template_h, self.template_b = self.template_gray.shape[:2]

    def detect(self, cell_image: np.ndarray) -> int:
        """
        Tæller antallet af kroner med Normalized Cross Correlation.
        Returnerer et heltal mellem 0 og 3.
        """
        if cell_image is None or cell_image.size == 0:
            return 0

            # konverterer billedbrikken fra farve (BGR) til gråtoner.
            # Fordele ved gråtoner:
            # - hurtigere fordi vi kun beregner på 1 billedkanal frem for 3
            # - mere robust overfor skiftende belysning og hvidbalance, da vi kun matcher på struktur og kontrast
        gray_cell = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

        # 3. Anvend Template Matching
        # Vi trækker skabelonen hen over billedet og beregner sammenligningen pixel for pixel.
        # TM_CCORR_NORMED (Normalized Cross-Correlation) beregner match-score (0.0 til  1.0),
        # mens self.mask fortæller funktionen, at den kun skal vurdere de pixels der ikke er gennemsigtige.
        res = cv2.matchTemplate(
            gray_cell, self.template_gray, cv2.TM_CCORR_NORMED, mask=self.mask
        )

        # 4. Find de Array-koordinater som overskrider threshold værdien
        # Vi filtrerer resultat-matricen og gemmer de (y, x) positioner, hvor match-scoren er højere end vores tærskelværdi.
        loc = np.where(res >= self.threshold)

        # Vi vender (y, x) til (x, y) koordinater og samler dem i en liste af punkter. Hvert punkt repræsenterer et potentielt krone-match.
        # loc: Indeholder ( [y1, y2, y3...], [x1, x2, x3...] ). Altså to lange lister.
        # [::-1]: med slicing at omvende rækkefølgen
        # * (Asterisk): Pakker tuplen ud, så vi har to separate lister i stedet for én tuple der indeholder to lister.
        # zip(): Tager den første x og parrer med den første y, tager den anden x og parrer med den anden y, osv. Den "lyner" dem sammen.
        # list(): Samler alle vores nye par i en pæn, itererbar liste: [(x1, y1), (x2, y2), (x3, y3)...].
        points = list(zip(*loc[::-1]))

        # 5. Non-Maximum Suppression (Overlapping Bounding Boxes)
        # Vi gør klar til at bygge firkanter (bounding boxes) omkring hvert fundet match-punkt.
        rectangles = []
        for x, y in points:
            # Hver boks defineres som [start_X, start_Y, bredde, højde].
            # Vi tilføjer bevidst den samme boks to gange. `cv2.groupRectangles` (længere nede)
            # kræver nemlig mindst 2 overlappende bokse for at godkende et fund, hvilket filtrerer støj fra.
            rectangles.append(
                [int(x), int(y), int(self.template_b), int(self.template_h)]
            )
            rectangles.append(
                [int(x), int(y), int(self.template_b), int(self.template_h)]
            )

        # Smelter klynger af overlappende bokse (fra samme krone) sammen til én enkelt boks.
        # eps=0.2 bestemmer afstanden for, hvornår bokse anses for at tilhøre samme gruppe (krone).
        rectangles, weights = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.2)

        # Antallet af bokse, vi har tilbage efter sammensmeltningen, er lig med antallet af faktiske kroner.
        crown_count = len(rectangles)

        # Kingdomino har maksimalt 3 kroner per tile
        return min(crown_count, 3)


if __name__ == "__main__":
    # --- Test af CrownDetector på Træning/Validering ---
    print("Initialiserer Template Matching (vha. high_res_crown.png)...")
    try:
        detector = CrownDetector(template_path="high_res_crown.png", threshold=0.95)
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

        board_crowns = 0

        for tile_file in sorted(board_folder.glob("*.jpg")):
            tile_img = cv2.imread(str(tile_file))

            if tile_img is None:
                continue

            crowns_found = detector.detect(tile_img)

            processed_tiles += 1
            board_crowns += crowns_found
            total_crowns += crowns_found

        print(f"  - {board_name}: {board_crowns} kroner")

    print("\nSamlet resultat:")
    print(f"  - Antal Brikker (Tiles) Scannet: {processed_tiles}")
    print(f"  - Estimeret Kongekroner Samlet : {total_crowns}")

    # Kan være at vi skal prøve at finjustere threshold,
    # eller alternativt prøve med hybridløsninger + farvemaskering
