
# Filen opdeler billder og gemmer dem som små tiles i mapper

import os
import cv2 as cv
from board_split import get_tiles


# Laves output mappe
input_folder = "KingDominoData"
output_folder = "KD_tiles"
os.makedirs(output_folder, exist_ok=True)


# Søges efter billeder
for file_name in os.listdir(input_folder):  
    if not file_name.lower().endswith("jpg"):     
        continue
    
    print(f"\n------ Processing image: {file_name} ------") 
    image_path = os.path.join(input_folder, file_name)
    image = cv.imread(image_path)
    if image is None:
        print(f"Can not read {file_name}")
        continue

    tiles = get_tiles(image) # 5 x 5 gitter
    
    # Laves en mappe per billdet
    board_name = os.path.splitext(file_name) [0]
    board_folder = os.path.join(output_folder, f"board_{board_name}")
    os.makedirs(board_folder, exist_ok=True)
    
    # Gem alle felter(tiles)
    for i, row in enumerate(tiles):
        for j, tile in enumerate(row):
            tile_path = os.path.join(board_folder, f"tile_{i}_{j}.jpg")
            tile_resized = cv.resize(tile, (128, 128))  # width x heigh
            cv.imwrite(tile_path, tile_resized)             # cv.imwrite(tile_path, tile)
            
    print(f"{file_name} devided and stored in {board_folder}")
            
    