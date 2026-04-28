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
from svm_final_test import test_svm_classifier
from create_crown_detection_results import run_template_matching
from bfs_algoritme import compute_score_from_csv



if __name__ == "__main__":
    
 
    # Extract 20-value vector for each tile: Feature_extract.py
    visualize_tile_and_histogram("KD_tiles/board_53/tile_1_2.jpg")
    visualize_tile_and_histogram("KD_tiles/board_29/tile_4_3.jpg")

    # Terrain Classification: svm_final_test.py
    test_svm_classifier()

    # Crown Detection
    run_template_matching()

    # Breadth First Search: bfs.py
    results = compute_score_from_csv("predictions_per_tile.csv")

    print("\n=== Board Scores ===")
    sorted_results = dict(
    sorted(results.items(), key=lambda x: int(x[0].split("_")[1]))
    )
    for board_name, score in sorted_results.items():
        print(f"{board_name}: {score}")



        
        
    
    

    
    