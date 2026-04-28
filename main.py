"""
King Domino Pipeline - Main Entry Point

This script serves as the orchestration hub for the King Domino terrain classification
and scoring pipeline. It imports and runs the SVM classifier evaluation.
"""

import cv2 as cv
import os

# importer fra .py-moduler
from board_split import get_tiles

from feature_extraction import visualize_tile_and_histogram
from bfs_algoritme import compute_score_from_csv
# from create_crown_detection_result_csv import # kun oprettelse af csv_results
from svm_final_test import test_svm_classifier



def main():
    
    dataset_folder = "KD_tiles"
    
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
    
        
        
    # Vis farve- histogram for nogle udvalgte tiles og deres histogram 
    
    visualize_tile_and_histogram("KD_tiles/board_53/tile_1_2.jpg")
    visualize_tile_and_histogram("KD_tiles/board_29/tile_4_3.jpg")

    

    # bfs.py
     
    results = compute_score_from_csv("ground_truth_per_tile.csv")

    print("\n=== Board Scores ===")
    sorted_results = dict(
    sorted(results.items(), key=lambda x: int(x[0].split("_")[1]))
    )
    for board_name, score in sorted_results.items():
        print(f"{board_name}: {score}")
    
    test_svm_classifier()
        
        
    
    

    
    