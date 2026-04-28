"""
Modul: Interaktivt annotationsværktøj til krone-detektion (Template Matching).

Dette modul leverer en grafisk brugergrænseflade (GUI), hvor man kan undersøge spilleplader,
markere (trække udsnit af) kroner fra HSV-farvebilleder og gemme disse som '.npy'-templates.
Det bruges primært til at opbygge det datasæt af templates, som `detect_crown.py` og systemet
senere benytter under evaluering.

Instrukser til opsætning:
- 'a' / 'd' : Naviger frem og tilbage mellem spillepladerne
- Træk en rektangel med musen : Gemmer markeringen som en ny skabelon (template)
- Klik på en GRØN firkant : Bekræfter markeringen og gemmer billedet af kronen
- ESC : Afslut programmet
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Hold-out sæt: Spilleplader der udelades fra alle træningstrin for at undgå datalækage ("data leakage").
TEST_BOARDS = {
    "board_55",

}

TM_THRESHOLD = (
    0.70  # Template matching tærskel (minimum tilladt match-score fra 0 til 1)
)
NMS_OVERLAP = (
    0.35  # Maksimalt tilladt overlap før en fundet krone betragtes som en dublet
)


def apply_nms(
    boxes: List[Tuple[int, int, int, int, float]], overlap_thresh: float
) -> List[Tuple[int, int, int, int, float]]:
    """
    Non-Maximum Suppression (NMS).
    Fjerner overlappende kasser (bounding boxes), så samme krone ikke detekteres flere gange.
    """
    if not boxes:
        return []

    # Sorter alle fundne kasser ud fra deres score, så de stærkeste matches vurderes først
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    retained = []

    for box in boxes:
        x1, y1, w, h, _ = box
        is_duplicate = False

        # Sammenlign med de kasser vi allerede har besluttet at beholde
        for bx, by, bw, bh, _ in retained:
            # Beregn det overlappende fællesareal
            ix = max(0, min(x1 + w, bx + bw) - max(x1, bx))
            iy = max(0, min(y1 + h, by + bh) - max(y1, by))

            # Hvis overlappet udgør en for stor del af kassens areal, er det en dublet
            if (ix * iy) / float(w * h) > overlap_thresh:
                is_duplicate = True
                break

        if not is_duplicate:
            retained.append(box)

    return retained


class InteractiveTemplateMatcher:
    """
    Styrer applikationens interaktive GUI via OpenCV, indlæsning af mapper,
    matching-sekvenserne og markering af nye kroner.
    """

    def __init__(self, script_dir: Path):
        self.script_dir = script_dir
        self.board_dir = self.script_dir / "KD_tiles"
        self.template_dir = self.script_dir / "template_hsv"
        self.kroner_dir = self.script_dir / "kroner_hsv"

        # Opret mapper hvis ikke de eksisterer (bruges til lagring af ML-skabeloner)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.kroner_dir.mkdir(parents=True, exist_ok=True)

        # Læs alle godkendte bræt-filer igennem
        self.board_files = self._find_boards()

        print(
            f"Indlæst {len(self.board_files)} spilleplader til annotation. "
            f"\n(Hold-out testsættet ekskluderes: {sorted(TEST_BOARDS)})"
        )

        # Tilstandsvariabler for brugergrænsefladen
        self.idx = 0
        self.board_orig = None
        self.board_hsv = None
        self.sel_start = None
        self.sel_end = None
        self.drawing = False
        self.found_boxes = []
        self.saved_boxes = set()

        # Titler på de to UI-vinduer
        self.window_hsv = "(HSV) only Hue, ignoring Saturation, ignoring Value"
        self.window_result = "Crowns found with correlation score"

    def _get_board_number(self, path_str: str) -> int:
        """Hjælpemetode, der trækker tallet ud af eksempelvis 'board_12', så spillebrættene kan sorteres numerisk."""
        for name in (Path(path_str).stem, Path(path_str).parent.name):
            try:
                return int(name.replace("board_", ""))
            except ValueError:
                continue
        return 0

    def _find_boards(self) -> List[str]:
        """Gennemgår filsystemet for alle trænings-tiles og returnerer en ordnet liste uden testsættet."""
        boards = []
        if self.board_dir.exists() and self.board_dir.is_dir():
            for i in self.board_dir.rglob("*.jpg"):
                num = self._get_board_number(str(i))
                # Filtrér test-sættet (forhindrer datalækage mens vi laver templates til træning)
                if num and f"board_{num}" in TEST_BOARDS:
                    boards.append(str(i))

        boards.sort(key=self._get_board_number)
        return boards

    def current_board_number(self) -> int:
        """Returnerer det aktuelle bræt's int-ID baseret på indeks."""
        num = self._get_board_number(self.board_files[self.idx])
        return num if num != 0 else Path(self.board_files[self.idx]).stem

    def load_board(self, idx: int):
        """Henter pladens billede fra filstien og laver en kopi i HSV, som fremhæver de gule (gold) kronfarver uanset lyssætning."""
        self.board_orig = cv2.imread(self.board_files[idx])
        # HSV (Hue, Saturation, Value) adskiller farvetone fra lysstyrke
        self.board_hsv = cv2.cvtColor(self.board_orig, cv2.COLOR_BGR2HSV)

    def draw_hsv(self):
        """Tegner det interaktive HSV-vindue, hvor brugeren markerer nye templates."""
        if self.board_hsv is None:
            return

        vis = self.board_hsv.copy()

        # Tegn markering mens billedet click & drages (når tegne-værktøjet "drawing" er aktivt)
        if self.drawing and self.sel_start and self.sel_end:
            cv2.rectangle(
                vis,
                self.sel_start,
                self.sel_end,
                color=(0, 120, 255),
                thickness=1,
            )

        # Skærm-overlay der fortæller brugeren hvor i processen vi er
        info_text = f"Board {self.current_board_number()}  [{self.idx + 1}/{len(self.board_files)}]"
        cv2.putText(
            img=vis,
            text=info_text,
            org=(6, 18),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.40,
            color=(0, 200, 255),
            thickness=1,
        )

        cv2.imshow(self.window_hsv, vis)

    def draw_results(self):
        """Tegner modellens's predictions i BGR original-billedet."""
        if self.board_orig is None:
            return

        resultat = self.board_orig.copy()

        # Tegn hver bounding box fundet af algoritmen
        for x, y, w, h, score in self.found_boxes:

            # Gør farven unik, hvis kronen blev markeret (klikket på)
            farve = (0, 255, 0)
            tekst = f"{score:.0%}"

            cv2.rectangle(
                img=resultat, pt1=(x, y), pt2=(x + w, y + h), color=farve, thickness=2
            )

            # Label
            cv2.putText(
                img=resultat,
                text=tekst,
                org=(x + 2, y + 12),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.35,
                color=farve,
                thickness=1,
            )

        cv2.imshow(self.window_result, resultat)

    def run_matching(self):
        """Selve hjertet af template matcheren. Trækker alle .npy-kroner hen over brættet."""
        self.found_boxes = []

        # Hent datasættet af numpy-skabeloner vi tidligere har udklippet
        templates = [np.load(f) for f in sorted(self.template_dir.glob("*.npy"))]
        templates = [t for t in templates if t is not None]

        if not templates or self.board_hsv is None:
            self.draw_results()
            return

        match_board = self.board_hsv.astype(np.float32)
        bh_board, bw_board = match_board.shape[:2]

        boxes = []
        for tmpl in templates:
            tmpl_f = tmpl.astype(np.float32)
            th, tw = tmpl_f.shape[:2]

            # En template skal selvsagt kunne rummes fysisk på selve match-brættet
            if th >= bh_board or tw >= bw_board:
                continue

            # Glid (Slide) template'n henover billedet via en normaliseret afstandsformel
            res = cv2.matchTemplate(match_board, tmpl_f, cv2.TM_CCOEFF_NORMED)

            # Find de pixels, hvor ligheden med templaten bryder over tærsklen
            locs = np.where(res >= TM_THRESHOLD)
            for y, x in zip(*locs):
                boxes.append((x, y, tw, th, float(res[y, x])))

        # Fjern overlap (samme krone set to gange)
        self.found_boxes = apply_nms(boxes, NMS_OVERLAP)

        self.draw_results()


    def save_template(self, x1: int, y1: int, x2: int, y2: int):
        """Gemmer det markerede udsnit som en template."""
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return

        crop = self.board_hsv[y1:y2, x1:x2]
        n = len(list(self.template_dir.glob("*.npy"))) + 1
        board_nr = self.current_board_number()
        sti = self.template_dir / f"template_{n:03d}_board{board_nr}_{x1}_{y1}.npy"

        np.save(sti, crop)
        print(f"Template gemt: {sti.name}  ({x2 - x1}×{y2 - y1} px)")
        self.run_matching()



    # ─── Callbacks ───────────────────────────────────────────────────────────
    def _mouse_hsv_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.sel_start = (x, y)
            self.sel_end = (x, y)
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.sel_end = (x, y)
            self.draw_hsv()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.sel_start and self.sel_end:
                self.save_template(
                    self.sel_start[0],
                    self.sel_start[1],
                    self.sel_end[0],
                    self.sel_end[1],
                )
            self.sel_start = None
            self.sel_end = None
            self.draw_hsv()



    # ─── Main Execution ──────────────────────────────────────────────────────
    def run(self):
        """Starter UI-loopet."""
        cv2.namedWindow(self.window_hsv, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_result, cv2.WINDOW_NORMAL)

        cv2.setMouseCallback(self.window_hsv, self._mouse_hsv_callback)

        self.load_board(self.idx)
        self.draw_hsv()
        self.run_matching()

        while True:
            tast = cv2.waitKey(50) & 0xFF

            if tast == 27:  # ESC → afslut
                break
            elif tast == ord("o"):  # o → annullér selektion
                self.drawing = False
                self.sel_start = None
                self.sel_end = None
                self.draw_hsv()
            elif tast in (83, ord("d")):  # d → næste board
                self.idx = (self.idx + 1) % len(self.board_files)
                self.saved_boxes = set()
                self.load_board(self.idx)
                self.draw_hsv()
                self.run_matching()
            elif tast in (81, ord("a")):  # a → forrige board
                self.idx = (self.idx - 1) % len(self.board_files)
                self.saved_boxes = set()
                self.load_board(self.idx)
                self.draw_hsv()
                self.run_matching()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = InteractiveTemplateMatcher(Path(__file__).parent.resolve())
    app.run()
