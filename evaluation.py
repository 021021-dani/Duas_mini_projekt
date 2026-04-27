

import os

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Importer vores moduler
from detect_crown import CrownDetector
from scoring import compute_board_score
from svm_clssifier import TEST_BOARDS, TileClssifier


def evaluate_pipeline():
    """
    Iputtersystemets hold-out billeder og udregner:
    Model Pointscore vs GT Pointscore = Mean Absolute Error
    """

    # 1. Indlæs Ground Truth CSV'er


    # Læs GT point score data
    gt_points = "Points_Spilleplade_GT.csv"
    print(f"\nIndlæser groundtruth point score csv: {gt_points}")
    gt_df = pd.read_csv(gt_points, sep=";")
    gt_scores = dict(zip(gt_df["board_name"], gt_df["point_score"]))

    # Læs GT crowns per tile data
    gt_crowns_csv = "GT_crowns_per_tile.csv"
    print(f"Indlæser groundtruth crowns per tile: {gt_crowns_csv}")
    gt_crowns_df = pd.read_csv(gt_crowns_csv, sep=";")
    # Opret et hurtigt dictionary til lookup: key=(board_name, tile_name) -> crowns
    gt_crowns_dict = {}
    for _, row in gt_crowns_df.iterrows():
        gt_crowns_dict[(row["Board"], row["Tile"])] = int(row["GT_Crowns"])

    # 2. Initialiser modeller (SVM og Template Matching)
    print("Instantierer SVM model")
    classifier = TileClssifier(features_csv="features.csv")
    if not classifier.is_fitted:
        print("Advarsel: SVM er ikke fitted. Kør svm_clssifier.py først.")
        return

    print("Instantierer template matching for at finde kroner")
    crown_detector = CrownDetector(templates_dir="template_hsv")

    dataset_folder = "KD_tiles"

    # Variabler til at gemme point resultater for MAE udregning
    predicted_points = []
    ground_truth_points = []

    # Variabler til crown-detektion MAE
    predicted_crowns = []
    ground_truth_crowns = []

    print(f"\nEvaluerer {len(TEST_BOARDS)} Hold-out testsæt:")
    print("-" * 50)

    # Test-boards er fx "board_1", "board_5", osv.
    for board_name in sorted(TEST_BOARDS, key=lambda x: int(x.split("_")[1])):
        board_id = board_name.split("_")[1]
        image_path = os.path.join(dataset_folder, f"{board_id}.jpg")

        if not os.path.isfile(image_path):
            print(
                f"Billede for {board_name} ikke fundet ({image_path}). Springer over."
            )
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Kunne ikke åbne billede: {image_path}")
            continue

        gt_score = gt_scores.get(board_name, None)
        if gt_score is None:
            print(f"Ingen Ground Truth fundet for {board_name}. Springer over.")
            continue

        # 3. Gitteropdeling (Grid Splitting)
        tiles_grid = get_tiles(image)

        # Opret numpy board matrix klar til `scoring.py`
        board_matrix = np.empty((5, 5), dtype=object)

        # 4. Inferens loop: For hver tile...
        for y in range(5):
            for x in range(5):
                tile_img = tiles_grid[y][x]

                # A) Multi-class Terrænklassificering
                terrain = classifier.classify(tile_img)

                # B) Krone-detektion vha Template Matching
                crowns = crown_detector.detect(tile_img)

                # Tilføj crowns metrics ved at lede dem op i GT dict
                tile_id = f"tile_{y}_{x}"
                gt_crown_val = gt_crowns_dict.get((board_name, tile_id))
                if gt_crown_val is not None:
                    predicted_crowns.append(crowns)
                    ground_truth_crowns.append(gt_crown_val)

                # Tilføj resultatet til vores matrix formateret på samme måde som fra CSV metoden
                board_matrix[y, x] = {"terrain": terrain, "crowns": crowns}

        # 5. Spillogik & BFS udregning
        predicted_score = compute_board_score(board_matrix, board_name, None)

        predicted_points.append(predicted_score)
        ground_truth_points.append(gt_score)

        diff = predicted_score - gt_score
        sign = "+" if diff > 0 else "" if diff == 0 else ""
        print(
            f"Plade: {board_name.ljust(10)} | Pred-Score: {predicted_score:2d} | GT-Score: {gt_score:2d} | Diff: {sign}{diff}"
        )

    # 6. Evaluerings-metrik udregning
    if predicted_points:
        mae_points = mean_absolute_error(ground_truth_points, predicted_points)
        print("-" * 50)
        print(
            f"OVERALL METRIK: Model Pointscore vs GT Pointscore = Mean Absolute Error (MAE): {mae_points:.2f}"
        )
        if predicted_crowns:
            mae_crowns = mean_absolute_error(ground_truth_crowns, predicted_crowns)
            print(
                f"OVERALL METRIK: Detekterede Kroner pt. Tile vs GT = Mean Absolute Error (MAE): {mae_crowns:.3f} kroner"
            )
        print("-" * 50)
    else:
        print("Ingen boards blev evalueret.")


if __name__ == "__main__":
    evaluate_pipeline()
