"""
Modul: Kronedetektion ved hjælp af Template Matching og Hyperparameter Tuning.

Dette modul tæller kongekroner på en tile vha. HSV-templates.
Inkluderer 5-Fold Cross-Validation for at optimere tærskelværdi (threshold) og overlap i NMS.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from create_templates import apply_nms

# Definer de spilleplader, der sorteres fra til test-sættet (Hold-out sæt)
TEST_BOARDS = {
    "board_1", "board_19", "board_23", "board_25", "board_29",
    "board_35", "board_39", "board_49", "board_53", "board_67", "board_67",
    "board_70",
}

# Opdel efter de 5 Fold-grupperinger til Cross-Validation
FOLD_MAPPING = {
    # Fold 0
    "board_4": 0,
    "board_8": 0,
    "board_20": 0,
    "board_24": 0,
    "board_34": 0,
    "board_38": 0,
    "board_42": 0,
    "board_46": 0,
    "board_48": 0,
    "board_52": 0,
    "board_65": 0,
    "board_72": 0,
    # Fold 1
    "board_2": 1,
    "board_6": 1,
    "board_18": 1,
    "board_22": 1,
    "board_28": 1,
    "board_32": 1,
    "board_36": 1,
    "board_40": 1,
    "board_51": 1,
    "board_55": 1,
    "board_58": 1,
    "board_62": 1,
    # Fold 2
    "board_10": 2,
    "board_14": 2,
    "board_11": 2,
    "board_15": 2,
    "board_26": 2,
    "board_30": 2,
    "board_41": 2,
    "board_44": 2,
    "board_57": 2,
    "board_61": 2,
    "board_64": 2,
    "board_68": 2,
    # Fold 3
    "board_3": 3,
    "board_7": 3,
    "board_17": 3,
    "board_21": 3,
    "board_27": 3,
    "board_31": 3,
    "board_43": 3,
    "board_47": 3,
    "board_50": 3,
    "board_54": 3,
    "board_59": 3,
    "board_63": 3,
    # Fold 4
    "board_9": 4,
    "board_13": 4,
    "board_12": 4,
    "board_16": 4,
    "board_33": 4,
    "board_37": 4,
    "board_45": 4,
    "board_56": 4,
    "board_60": 4,
    "board_66": 4,
    "board_69": 4,
}


class CrownDetector:
    """
    Klassificerer og tæller kroner for en enkelt tile vha. lokalt gemte numpy (.npy) templates.
    """

    def __init__(self, templates_dir="template_hsv", threshold=0.70, nms_overlap=0.35):
        self.threshold = threshold
        self.nms_overlap = nms_overlap
        self.templates = []

        # Indlæs og gem alle vores numpy-templates i forvejen
        templates_path = Path(templates_dir)
        if templates_path.exists():
            for f in sorted(templates_path.glob("*.npy")):
                # Uddrag board-nummer fra filnavnet, f.eks. "template_001_board12_10_20.npy"
                parts = f.name.split("_")
                if len(parts) >= 3 and parts[2].startswith("board"):
                    board_num = parts[2].replace("board", "")
                    # Filtrér test-sættet (forhindrer datalækage)
                    if f"board_{board_num}" in TEST_BOARDS:
                        continue

                t = np.load(f)
                if t is not None:
                    self.templates.append(t.astype(np.float32))

    def detect(self, cell_image: np.ndarray) -> int:
        """
        Input: BGR-billede af feltet (fx. en 100x100 pixel tile).
        Output: Heltal med antal fundne kroner vha. template matching og NMS.
        """
        # Afbryd, hvis billedet mangler, eller hvis ingen templates er loadet
        if cell_image is None or cell_image.size == 0 or not self.templates:
            return 0

        # Konverter tile-billedet til HSV-farverum (bedre til guld/gul under skiftende lys)
        cell_hsv = cv2.cvtColor(cell_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        bh, bw = cell_hsv.shape[:2]

        boxes = []
        # Udfør cv2.matchTemplate henover billedfirkanten med hver template-fil
        for tmpl_f in self.templates:
            th, tw = tmpl_f.shape[:2]

            # Spring templaten over, hvis den f.eks er større end selve tilen
            if th >= bh or tw >= bw:
                continue

            res = cv2.matchTemplate(cell_hsv, tmpl_f, cv2.TM_CCOEFF_NORMED)

            # Find (x, y) pixellokationer med en score, der er højere end vores threshold
            locs = np.where(res >= self.threshold)

            for y, x in zip(*locs):
                boxes.append((x, y, tw, th, float(res[y, x])))

        # Filtrér eventuelle dubletter (samme krone ramt af flere skabeloner)
        retained_boxes = apply_nms(boxes, self.nms_overlap)

        # Spillereglerne siger maks. 3 kroner pr. terrænfelt
        return min(len(retained_boxes), 3)


if __name__ == "__main__":
    print("--- Hyperparameter Tuning: 5-Fold CV for CrownDetector ---")

    # Læs den oprindelige CSV med pandas, for at få det totale antal kroner (facit) for hvert board
    gt_df = pd.read_csv("PointScore_ground_truth.csv", sep=";")
    gt_dict = {row["board_name"]: row["total_crowns"] for _, row in gt_df.iterrows()}

    # Læs tile-niveau GT kroner (facit) for hver brik
    gt_crowns_df = pd.read_csv("GT_crowns_per_tile.csv", sep=";")
    gt_crowns_dict = {
        (row["Board"], row["Tile"]): int(row["GT_Crowns"])
        for _, row in gt_crowns_df.iterrows()
    }

    tiles_dir = Path("KD_tiles")
    if not tiles_dir.exists():
        print(f"Mappe '{tiles_dir}' ikke fundet.")
        exit(1)

    # --- 1. Opsæt Grid Search ---
    # Vi tuner over de tre værdier på Threshold og Overlap-parametre
    param_grid = {
        "threshold": [0.73],
        "nms_overlap": [0.35],
    }

    """Resultater:
    """

    best_mae = float("inf")
    best_params = {}

    from itertools import product

    combinations = list(product(param_grid["threshold"], param_grid["nms_overlap"]))

    print(f"Tester {len(combinations)} parameter-kombinationer (med GridSearchCV)\n")

    for thresh, overlap in combinations:
        detector = CrownDetector(
            templates_dir="template_hsv", threshold=thresh, nms_overlap=overlap
        )

        # Et dictionary til at opbevare fejl for hvert af vores 5 CV folds
        fold_errors = {i: [] for i in range(5)}

        # Læs mappens spilleplader systematisk til valideringen
        for board_folder in tiles_dir.iterdir():
            if not board_folder.is_dir():
                continue

            board_name = board_folder.name

            if board_name in TEST_BOARDS:
                continue  # Test-sættet udelades fra træning for at undgå datalækage ("data leakage")
            if board_name not in FOLD_MAPPING:
                continue

            fold = FOLD_MAPPING[board_name]
            true_total_crowns = gt_dict.get(board_name, 0)

            predicted_board_crowns = 0

            # Tilføj antal fundne kroner til pladens samlede antal
            for tile_file in board_folder.glob("*.jpg"):
                tile_orig = cv2.imread(str(tile_file))
                predicted_board_crowns += detector.detect(tile_orig)

            # Beregn den absolutte afvigelse for denne model
            fejl = abs(predicted_board_crowns - true_total_crowns)
            fold_errors[fold].append(fejl)

        # Beregn Mean Absolute Error (MAE) for hver parameter-kombination
        fold_maes = [np.mean(errs) if errs else 0 for errs in fold_errors.values()]
        avg_cv_mae = np.mean(fold_maes)

        print(
            f"Params (Threshold={thresh:.2f}, Overlap={overlap:.2f}) -> CV MAE: {avg_cv_mae:.2f} kroner galt pr. plade"
        )

        # Træn kun på dem der har færrest "gæt" forkert
        if avg_cv_mae < best_mae:
            best_mae = avg_cv_mae
            best_params = {"threshold": thresh, "nms_overlap": overlap}

    print(
        f"Gennemsnitlig bedste parameter-sæt: {best_params} med MAE på {best_mae:.2f} kroner i forskel."
    )

    print(
        "\n Opretter test-rapporter: 'evaluation_boards.csv' og 'evaluation_tiles.csv'"
    )

    processed_tiles = 0
    estimated_total_crowns = 0

    board_results = []
    tile_results = []

    # Hjælpefunktion til at sortere spillepladerne numerisk (1, 2, 3... i stedet for 1, 10, 11)
    def board_sort_key(name):
        try:
            return int(name.replace("board_", ""))
        except ValueError:
            return 0

    # Iterer over hele datasættet og foretag inference over de rigtige labels
    for board_folder_name in sorted(os.listdir(tiles_dir), key=board_sort_key):
        board_folder = os.path.join(tiles_dir, board_folder_name)

        if not os.path.isdir(board_folder):
            continue

        board_name = board_folder_name

        # Markér explicit om rækken hører til test-sættet
        is_test_board = board_name in TEST_BOARDS

        board_crowns = 0
        true_board_crowns = gt_dict.get(board_name, None)

        for tile_file_name in sorted(os.listdir(board_folder)):
            if not tile_file_name.endswith(".jpg"):
                continue

            tile_file = os.path.join(board_folder, tile_file_name)
            tile_img = cv2.imread(tile_file)

            if tile_img is None:
                continue

            # Lav det endelige inference tjek
            crowns_found = detector.detect(tile_img)

            processed_tiles += 1
            board_crowns += crowns_found
            estimated_total_crowns += crowns_found

            # Bevar index så det matcher template_matching.py format, f.eks. "tile_2_3" fra "tile_2_3.jpg"
            tile_idx = Path(tile_file_name).stem

            true_tile_crowns = gt_crowns_dict.get((board_name, tile_idx), "N/A")

            tile_results.append(
                {
                    "Board": board_name,
                    "Tile": tile_idx,
                    "Is_Test_Set": is_test_board,
                    "GT_Crowns": true_tile_crowns,
                    "Detected_Crowns": crowns_found,
                    "Error": (crowns_found - true_tile_crowns)
                    if true_tile_crowns != "N/A"
                    else "N/A",
                }
            )

        board_results.append(
            {
                "Board": board_name,
                "Is_Test_Set": is_test_board,
                "GT_Crowns": true_board_crowns,
                "Detected_Crowns": board_crowns,
                "Error": (board_crowns - true_board_crowns)
                if true_board_crowns is not None
                else "N/A",
            }
        )

    print(f"\n {len(board_results)} spilleplader og {len(tile_results)} terrain-tiles.")

    # Eksporter tabeller til CSV med semikolon separator
    board_df = pd.DataFrame(board_results)
    tile_df = pd.DataFrame(tile_results)

    board_df.to_csv("evaluation_boards.csv", index=False, sep=";")
    tile_df.to_csv("evaluation_tiles.csv", index=False, sep=";")

    # --- Print Evaluation Results (Merged from count_crowns.py) ---
    # Presents training results:
    training_set_df = tile_df[tile_df["Is_Test_Set"] == False]
    training_GT_Crowns = pd.to_numeric(
        training_set_df["GT_Crowns"], errors="coerce"
    ).sum()
    training_Detected_Crowns = pd.to_numeric(
        training_set_df["Detected_Crowns"], errors="coerce"
    ).sum()
    training_Error = pd.to_numeric(training_set_df["Error"], errors="coerce").sum()

    # print("\n--- Training Set Only ---")
    # print(f"Training GT Crowns: {training_GT_Crowns}")
    # print(f"Training Detected Crowns: {training_Detected_Crowns}")
    # print(f"Training Error: {training_Error}")

    # Presents test set results:
    test_set_df = tile_df[tile_df["Is_Test_Set"] == True]
    test_GT_Crowns = pd.to_numeric(test_set_df["GT_Crowns"], errors="coerce").sum()
    test_Detected_Crowns = pd.to_numeric(
        test_set_df["Detected_Crowns"], errors="coerce"
    ).sum()
    test_Error = pd.to_numeric(test_set_df["Error"], errors="coerce").sum()

    print("\n--- Test Set Only ---")
    print(f"Test GT Crowns: {test_GT_Crowns}")
    print(f"Test Detected Crowns: {test_Detected_Crowns}")
    print(f"Test Error: {test_Error}")
