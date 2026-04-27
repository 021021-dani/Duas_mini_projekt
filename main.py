
import cv2 as cv
import pandas as pd
import os
from board_split import get_tiles
from feature_extraction import extrac_hsv_histogram, process_all_tiles, visualize_tile_and_histogram
from scoring import compute_score_from_csv


def main():
    
    dataset_folder = r"C:\Duas_mini_projekt\KD_tiles"
    
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
    
    # Gem alle features i CSV inkl. ground truth labels
    
    process_all_tiles(
        tiles_root_folder="KD_tiles",
        output_csv="features.csv",
        ground_truth_csv="Labels_ground_truth.csv"
    )
        
        
    # Vis farve- histogram for nogle udvalgte tiles og deres histogram 
    
    #visualize_tile_and_histogram("KD_tiles/board_1/tile_0_0.jpg")
    #visualize_tile_and_histogram("KD_tiles/board_49/tile_1_2.jpg")
    visualize_tile_and_histogram("KD_tiles/board_6/tile_1_2.jpg")
    visualize_tile_and_histogram("KD_tiles/board_46/tile_4_4.jpg")
    

    # scoring.py
    
    #results = compute_score_from_csv("features_with_crowns.csv")    
    
    #print("\n=== Board Scores ===")
    #for board_name, score in results.items():
        #print(f"{board_name}: {score}")
        
        
        
        
    
    

    
    