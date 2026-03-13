import cv2 as cv
import os

def main():

    print("\n======= Loading KD Images =====")

    dataset_folder = r"/Users/mamali/Desktop/Ai_aau/Projekter/Duas_mini_projekt/KingDominoData"

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

if __name__ == "__main__":
    main()