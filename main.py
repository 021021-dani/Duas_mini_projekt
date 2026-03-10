import cv2 as cv
import os
import numpy as np

def main():
    
    print("+-------------------------------+")
    print("| King Domino points calculator |")
    print("+-------------------------------+")
    
    print("\n======= Loadin KD Images =====")
    folder = r"/Users/mamali/Desktop/Ai_aau/Projekter/Mini_projekt2/KingDominoData"
    for file_name in sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0])): # Ordner Rækkefølge af billeder
        if file_name.lower().endswith(".jpg"):
            image_path = os.path.join(folder, file_name)
            print("Processing:", file_name)
            
            if not os.path.isfile(image_path):
                print("Image not found")
            image = cv.imread(image_path)
            
if __name__ =="__main__":
    main() 
