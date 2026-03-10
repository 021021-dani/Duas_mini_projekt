import cv2 as cv
import os

def main():

    print("\n======= Loadin KD Images =====")
    
    dataset_folder = r"/Users/mamali/Desktop/Ai_aau/Projekter/Mini_projekt2/KingDominoData"
    for file_name in sorted(os.listdir(dataset_folder), key=lambda x: int(x.split(".")[0])): # Ordner Rækkefølge af billeder
        if file_name.lower().endswith(".jpg"):
            image_name = os.path.join(dataset_folder, file_name)
            print("Processing:", file_name)
            
            if not os.path.isfile(image_name):
                print("Image not found")
            image = cv.imread(image_name)
            
if __name__ =="__main__":
    main() 

