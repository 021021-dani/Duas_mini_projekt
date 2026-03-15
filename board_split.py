
# Filen Indeholder kun funktionaliteten til at opdele et billede i 5x5-tiles hvilket gør koden mere modulær og genbrugelig.

import cv2 as cv


def get_tiles(image):
    """
    Hent højde og bredde fra billedet uden farvekanaler og
    Deler et billede op i et gitter af mindre felter (tiles) 
    """
    
    h, w = image.shape[:2]  
    tile_h = h // 5         
    tile_w = w // 5          

    tiles = []      
    
    for y in range(5):
        row = []
        for x in range(5):

            tile = image[y*tile_h:(y+1)* tile_h,      # Udskærer en del af billedet (en tile) baseret på beregnede størrelser
                        x*tile_w:(x+1)* tile_w]
            
            row.append(tile)     

        tiles.append(row)       
        
    return tiles            