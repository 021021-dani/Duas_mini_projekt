import csv
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def extrac_hsv_histogram(tile):
    """
    Udtræk farvehistogram fra en tile i HSV-farverum.
    Returnerer en numpy array med 20 værdier.
    """

    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)  # Konverter fra RGB til HSV

    # Beregn histogram for hver kanal ([0] = kanal, None = ingen maske,
    # [10] = antal bins, [0,180] = værdiområde for HSV

    h_hist = cv.calcHist([hsv], [0], None, [10], [0, 180])
    s_hist = cv.calcHist([hsv], [1], None, [5], [0, 256])
    v_hist = cv.calcHist([hsv], [2], None, [5], [0, 256])

    # Normaliser så lysstryke ikke dominerer (alle værdier mellem 0 og 1)

    h_hist = cv.normalize(h_hist, h_hist).flatten()
    s_hist = cv.normalize(s_hist, s_hist).flatten()
    v_hist = cv.normalize(v_hist, v_hist).flatten()

    # Sæt de tre histogrammer sammen til én vektor med 20 tal
    features = np.concatenate([h_hist, s_hist, v_hist])

    return features


def process_all_tiles(
    tiles_root_folder, output_csv, ground_truth_csv="Labels_ground_truth.csv"
):
    """
    Gennemgår alle board-mapper og deres tiles,
    udtrækker features og gemmer i en CSV-fil.
    """

    # Indlæs ground truth labels
    ground_truth = {}
    if os.path.exists(ground_truth_csv):
        with open(ground_truth_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            headers = next(reader)
            # Find the header indices for each tile mapping
            for row in reader:
                if not row or len(row) < 2:
                    continue
                board_name = row[0].strip()
                ground_truth[board_name] = {}
                for col_idx in range(1, len(headers)):
                    tile_name = headers[col_idx].strip()
                    if tile_name and col_idx < len(row):
                        ground_truth[board_name][tile_name] = row[col_idx].strip()
    else:
        print(
            f"BEMÆRK: Kunne ikke finde '{ground_truth_csv}'. Alle labels bliver sat til 'Unknown'."
        )

    # Kolonner til CSV-filen (10 hue, 5 sat, 5 val)

    header = (
        [f"hue_{i}" for i in range(10)]
        + [f"sat_{i}" for i in range(5)]
        + [f"val_{i}" for i in range(5)]
        + ["label", "tile_file", "board_name"]
    )

    rows = []  # Samler alle rækker inden de skrives til CSV

    # Find alle board mapper og sorter dem numerisk (1, 2, 3...)
    ordered_boards = [
        d
        for d in os.listdir(tiles_root_folder)
        if os.path.isdir(os.path.join(tiles_root_folder, d)) and d.startswith("board_")
    ]
    ordered_boards.sort(key=lambda x: int(x.split("_")[1]))

    # Gå gennem alle board-mapper i den specifikke rækkefølge
    for board_name in ordered_boards:
        board_path = os.path.join(tiles_root_folder, board_name)

        # Spring over hvis det ikke er en mappe

        if not os.path.isdir(board_path):
            continue
        print(f"Treats: {board_name}")

        # Gå gennem alle tiles-billeder inde i board-mappen

        for tile_file in sorted(os.listdir(board_path)):
            if not tile_file.lower().endswith(".jpg"):
                continue

            # Byg den fulde sti til tile-billedet

            tile_path = os.path.join(board_path, tile_file)

            tile = cv.imread(tile_path)  # Indlæse billedet med cv2

            # Spring over hvis billedet ikke kunne læses

            if tile is None:
                print(f"  Can not read: {tile_file}")
                continue

            # Udtræk de 20 histogram-værdier fra tilen

            features = extrac_hsv_histogram(tile)

            # Find det rigtige label fra ground truth
            tile_key = tile_file.split(".")[0]  # Dvs. fra "tile_0_0.jpg" til "tile_0_0"
            label = "Unknown"
            if board_name in ground_truth and tile_key in ground_truth[board_name]:
                label_val = ground_truth[board_name][tile_key]
                if label_val:  # Sørg for at den ikke er tom
                    label = label_val

            # Bygge en række til CSV, runder 4 decimaler + label, filnavn og board_name

            row = list(np.round(features, 4)) + [label, tile_file, board_name]
            rows.append(row)

    # Skrives alle rækker til CSV-filen på en gang
    with open(
        output_csv, "w", newline=""
    ) as f:  #  f variablen bruge  i stedet for at skrive hele filnavnet igen.
        # det håndterer automatisk kommaer og anførselstegn korrekt.
        writer = csv.writer(f)

        writer.writerow(header)  # Første linje = klonnenavne
        writer.writerows(rows)  # Alle efterfølgende rækker = data

    print(f"\nFinished! {len(rows)} tiles save in: {output_csv}")


def draw_histogram(ax, hist, colors, title, labels):
    """Tegner ét histogram med markering af den højeste søjle"""

    # Tegn alle søjler

    ax.bar(range(len(hist)), hist, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xticks(range(len(hist)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Normaliseret frekvens")

    # Find og marker den højeste søjle med fed kant og værdi

    max_idx = np.argmax(hist)
    ax.bar(
        max_idx, hist[max_idx], color=colors[max_idx], edgecolor="black", linewidth=1.5
    )
    ax.text(
        max_idx,
        hist[max_idx] + 0.02,
        f"{hist[max_idx]:.2f}",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )


def visualize_tile_and_histogram(tile_path):
    """
    Viser tile-billedet og dets HSV-histogram side om side
    for at se sammenhængen mellem billede og Histogram.
    """

    # Indlæs billedet fra stien

    tile = cv.imread(tile_path)
    if tile is None:
        print("Can not read:", tile_path)
        return

    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)

    tile_rgb = cv.cvtColor(tile, cv.COLOR_BGR2RGB)

    # Beregn og normaliser histogrammerne

    h_hist = cv.normalize(cv.calcHist([hsv], [0], None, [10], [0, 180]), None).flatten()
    s_hist = cv.normalize(cv.calcHist([hsv], [1], None, [5], [0, 256]), None).flatten()
    v_hist = cv.normalize(cv.calcHist([hsv], [2], None, [5], [0, 256]), None).flatten()

    hue_colors = [
        "#FF4444",
        "#FF8800",
        "#AACC00",
        "#44BB00",
        "#00AA88",
        "#0088FF",
        "#4444FF",
        "#8800FF",
        "#CC0088",
        "#FF4444",
    ]
    sat_colors = ["#BBBBBB", "#8899BB", "#4477BB", "#1144AA", "#001188"]
    val_colors = ["#111111", "#555555", "#999999", "#CCCCCC", "#FFFFFF"]

    # Labels til x-akserne

    hue_labels = [
        "rød",
        "org",
        "gul\ngrøn",
        "grøn",
        "cyan",
        "blå",
        "mørk\nblå",
        "lilla",
        "pink",
        "rød",
    ]
    sat_labels = ["grå\nmat", "lav", "middel", "høj", "stærk\nfarve"]
    val_labels = ["meget\nmørk", "mørk", "middel", "lys", "meget\nlys"]

    # Lav et vindue med 4 plots side om side 1 tile-billede + 3 histogrammer
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Sæt filnavnet som titel øverst på vinduet
    fig.suptitle(os.path.basename(tile_path), fontsize=13)

    # Vis selve tile-billedet som første plot

    axes[0].imshow(tile_rgb)
    axes[0].set_title("Tile")
    axes[0].axis("off")

    # Tegn de tre histogrammer med hjælpefunktionen

    draw_histogram(axes[1], h_hist, hue_colors, "Hue (farvetone)", hue_labels)
    draw_histogram(axes[2], s_hist, sat_colors, "Saturation (mætning)", sat_labels)
    draw_histogram(axes[3], v_hist, val_colors, "Value (lysstyrke)", val_labels)

    plt.tight_layout()
    plt.show()
