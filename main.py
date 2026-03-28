import cv2 as cv
import os
from board_split import get_tiles
from feature_extraction import extrac_hsv_histogram, process_all_tiles, visualize_tile_and_histogram

def main():
    dataset_folder = r"/Users/mamali/Desktop/Ai_aau/Projekter/Duas_mini_projekt/KingDominoData"
    print("\n====== Searching for images =======")
    files = [
        f for f in os.listdir(dataset_folder)
        if f.lower().endswith(".jpg") and f.split(".")[0].isdigit()
    ]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    for file_name in files:
        image_path = os.path.join(dataset_folder, file_name)
        print("Loading image:", file_name)
        if not os.path.isfile(image_path):
            print("Image not found")
            continue
        image = cv.imread(image_path)
        if image is None:
            print("Could not load image:", file_name)
            continue
        tiles = get_tiles(image)
        print("Tiles rows:", len(tiles))
        print("Tiles columns:", len(tiles[0]))

if __name__ == "__main__":
    
    # Gem alle features i CSV
    
    process_all_tiles(
        tiles_root_folder="KD_tiles",
        output_csv="features.csv"
    )

    # Vis histogram for nogle udvalgte tiles
    visualize_tile_and_histogram("KD_tiles/board_1/tile_0_0.jpg")
    visualize_tile_and_histogram("KD_tiles/board_1/tile_1_0.jpg")
    visualize_tile_and_histogram("KD_tiles/board_1/tile_2_0.jpg")