import cv2 as cv
import os

import os
import cv2 as cv

def main():

    print("\n======= Loading KD Images =====")
    
    dataset_folder = r"/Users/mamali/Desktop/Ai_aau/Projekter/Mini_projekt2/Duas_mini_projekt/KingDominoData"
    

    files = [
        f for f in os.listdir(dataset_folder)
        if f.lower().endswith(".jpg") and f.split(".")[0].isdigit()
    ]

    # Sorter efter tallet i filnavnet
    files = sorted(files, key=lambda x: int(x.split(".")[0]))

    for file_name in files:
        image_name = os.path.join(dataset_folder, file_name)
        print("Processing:", file_name)

        if not os.path.isfile(image_name):
            print("Image not found")
            continue

        image = cv.imread(image_name)

        if image is None:
            print("Could not load image:", file_name)
            continue

if __name__ == "__main__":
    main()

