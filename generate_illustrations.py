from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from board_split import get_tiles
from detect_crown import FOLD_MAPPING, TEST_BOARDS, CrownDetector, apply_nms

# Import from existing project files
from svm_clssifier import TileClssifier
from template_matching import NMS_OVERLAP as TEMPLATE_MATCHING_NMS_OVERLAP

# Create an output folder for illustrations
OUT_DIR = Path("aflevering/illustrations")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_template_matching_illustration():
    """1. An illustration of the template matching process."""
    print("Generating template matching and NMS illustrations...")

    # We load the case example tile from board_7, tile_0_0
    # Note: Requires KingDominoData/49.jpg to exist
    board_path = Path("KingDominoData/7.jpg")
    if not board_path.exists():
        print("Board 1 not found, skipping template matching illustration.")
        return

    board_img = cv2.imread(str(board_path))
    tiles = get_tiles(board_img)
    tile_img = tiles[0][0]  # case_example: board_7, tile_0_0

    detector = CrownDetector(templates_dir="template_hsv")
    if not detector.templates:
        print("No templates found.")
        return

    def collect_template_matches(tile_hsv: np.ndarray):
        matches = []
        bh, bw = tile_hsv.shape[:2]

        for path in sorted(Path("template_hsv").glob("*.npy")):
            template = np.load(path)
            if template is None:
                continue

            template = template.astype(np.float32)
            th, tw = template.shape[:2]
            if th >= bh or tw >= bw:
                continue

            res = cv2.matchTemplate(tile_hsv, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            matches.append(
                {
                    "path": path,
                    "template": template,
                    "score": float(max_val),
                    "loc": max_loc,
                    "size": (tw, th),
                    "heatmap": res,
                }
            )

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches

    # Convert tile to HSV
    tile_hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Perform matching across all available templates and keep the strongest real matches
    matches = collect_template_matches(tile_hsv)
    if not matches:
        print("No usable template matches found.")
        return

    best_match = matches[0]
    template = best_match["template"]
    res = best_match["heatmap"]
    max_loc = best_match["loc"]
    max_val = best_match["score"]

    # 1. Visualization: The Process
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Search Image (BGR)")
    axs[0].axis("off")

    heatmap = axs[1].imshow(res, cmap="inferno")
    axs[1].set_title("Correlation Heatmap")
    axs[1].axis("off")
    color_scale = plt.cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap="inferno")
    color_scale.set_array([])
    fig.colorbar(color_scale, ax=axs[1], fraction=0.046, pad=0.04)

    best_match_img = tile_img.copy()
    h, w = template.shape[:2]
    cv2.rectangle(
        best_match_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2
    )
    axs[2].imshow(cv2.cvtColor(best_match_img, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Best Match Location")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "1_template_matching_process.png", dpi=200)
    plt.close()

    # 2. Visualization: NMS Before and After
    # Use the strongest real template matches so every box reflects an actual
    # template that participated in locating the crown on this tile.
    boxes = []
    nms_source_matches = matches[:5]
    for match in nms_source_matches:
        x, y = match["loc"]
        bw, bh = match["size"]
        boxes.append((x, y, bw, bh, match["score"]))

    img_before = tile_img.copy()
    print("\nBounding boxes before NMS:")
    for idx, (x, y, bw, bh, score) in enumerate(boxes, start=1):
        print(f"  #{idx}: x={x}, y={y}, w={bw}, h={bh}, score={score:.3f}")
        cv2.rectangle(img_before, (x, y), (x + bw, y + bh), (0, 0, 255), 1)
        label = f"{idx}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
        )
        label_x = x + max((bw - text_w) // 2, 0)
        label_y = y + max((bh + text_h) // 2, text_h)
        cv2.putText(
            img_before,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
        )

    retained_boxes = apply_nms(boxes, TEMPLATE_MATCHING_NMS_OVERLAP)
    img_after = tile_img.copy()
    print("Bounding boxes after NMS:")
    for idx, (x, y, bw, bh, score) in enumerate(retained_boxes, start=1):
        print(f"  #{idx}: x={x}, y={y}, w={bw}, h={bh}, score={score:.3f}")
        cv2.rectangle(img_after, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        label = f"{idx}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
        )
        label_x = x + max((bw - text_w) // 2, 0)
        label_y = y + max((bh + text_h) // 2, text_h)
        cv2.putText(
            img_after,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
        )

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    axs[0].set_title(
        f"Before NMS (Raw Matches, overlap={TEMPLATE_MATCHING_NMS_OVERLAP:.2f})"
    )
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    axs[1].set_title(
        f"After NMS (Filtered, overlap={TEMPLATE_MATCHING_NMS_OVERLAP:.2f})"
    )
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "2_multiple_bounding_boxes_nms.png", dpi=200)
    plt.close()


def generate_dataset_split_illustration():
    """3. An illustration of the dataset split strategy."""
    print("Generating dataset split illustration...")
    # Map folds to numbers
    data = []
    # 36 total unique boards, 6 test
    for b in range(1, 73):
        b_name = f"board_{b}"
        if b_name in TEST_BOARDS:
            data.append({"Board": b, "Fold": "Hold-out (Test)"})
        elif b_name in FOLD_MAPPING:
            data.append({"Board": b, "Fold": f"Fold {FOLD_MAPPING[b_name] + 1}"})

    df = pd.DataFrame(data).dropna()

    plt.figure(figsize=(10, 4))
    sns.countplot(
        data=df,
        x="Fold",
        palette="viridis",
        order=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Hold-out (Test)"],
    )
    plt.title("Dataset Split Strategy: 5-Fold CV vs Hold-out Set")
    plt.ylabel("Number of Boards")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "3_dataset_split_strategy.png", dpi=200)
    plt.close()


def generate_confusion_matrix_illustration():
    """4. An illustration of the terrain classification confusion matrix."""
    print("Generating terrain classification confusion matrix...")
    features_csv = Path("features.csv")
    if not features_csv.exists():
        print("features.csv not found. Skip.")
        return

    df = pd.read_csv(features_csv)

    # Train SVM on non-test
    train_df = df[~df["board_name"].isin(TEST_BOARDS)].copy()
    test_df = df[df["board_name"].isin(TEST_BOARDS)].copy()

    if test_df.empty:
        print("No test data found in features.csv. Skip.")
        return

    # Init classifier
    classifier = TileClssifier("features.csv")
    if not classifier.is_fitted:
        print("SVM not fitted.")
        return

    # Get true vs pred
    X_test = test_df.iloc[:, :-3].to_numpy(dtype=float)
    y_test_true = test_df["label"].to_numpy(dtype=str)
    y_test_pred = classifier.svm_model.predict(X_test)

    labels = classifier.svm_model.classes_
    cm = confusion_matrix(y_test_true, y_test_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix on Hold-out Set (SVM)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "4_terrain_classification_cm.png", dpi=200)
    plt.close()


def generate_crowns_scatter_illustration():
    """5. Comparing predicted crowns versus ground-truth crowns per tile."""
    print("Generating crowns comparison scatter plot...")

    gt_csv = Path("GT_crowns_per_tile.csv")
    eval_csv = Path("evaluation_tiles.csv")

    if not gt_csv.exists() or not eval_csv.exists():
        print("Missing GT_crowns_per_tile.csv or evaluation_tiles.csv. Skip.")
        return

    # Load evaluation logic to get predictions
    eval_df = pd.read_csv(eval_csv, sep=";")
    gt_df = pd.read_csv(gt_csv, sep=";")

    # Join on Board and Tile while keeping the predicted and ground-truth columns separate
    merged = pd.merge(
        eval_df,
        gt_df,
        on=["Board", "Tile"],
        how="inner",
        suffixes=("_pred", "_gt"),
    )

    # Only keep hold out test set to remain consistent
    merged = merged[merged["Is_Test_Set_pred"] == True]

    if merged.empty:
        print("No test set tiles found in joined dataframe.")
        return

    # To avoid severe point overlapping, we add a bit of jitter to the scatter plot
    plt.figure(figsize=(6, 6))
    sns.stripplot(
        x="GT_Crowns_gt", y="Detected_Crowns", data=merged, jitter=0.2, alpha=0.6
    )

    # Draw perfect prediction line
    plt.plot([-0.5, 3.5], [-0.5, 3.5], "r--", label="Perfect Prediction (MAE=0)")

    plt.title("Predicted vs Ground Truth Crowns per Tile (Test Set)")
    plt.xlabel("Ground Truth Crowns")
    plt.ylabel("Protected Crowns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "5_predicted_vs_gt_crowns.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    generate_template_matching_illustration()
    generate_dataset_split_illustration()
    generate_confusion_matrix_illustration()
    # generate_crowns_scatter_illustration()

    print(f"\nAll illustrations generated and saved to {OUT_DIR.resolve()}!")
