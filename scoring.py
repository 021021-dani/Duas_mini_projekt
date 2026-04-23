
""" Ansvar:
# - Opbygge 5x5 array(board) fra CSV
# - Anvendes BFS til at finde regioner
# - Beregne Kingdomino score(antal Kongekroner)+ antal sammenhængende områder! """

import pandas as pd
import numpy as np
import cv2 as cv
from collections import deque 

# --------------------------------------- 
# 1. Hjælpefunktion: parse tile position
# --------------------------------------

def parse_tile_position(tile_name):
    """  
    Ekstraher row, cell fra f.eks. tile_2_3.jpg til (2,3)!
    """
    
    name = tile_name.split(".")[0]
    _, row_idx, cold_idx = name.split("_")
    return int(row_idx), int(cold_idx)

# ----------------------
# 2. Byg board (5 x 5)
# ----------------------

def build_board_matrix(board_df):
    """ 
    Konverter Dataframe til ---> 5 x 5 numpy array,
    Hver Celle: {"terrain": str, "crowns": int} !
    """
    
     # Opret tomt 5x5 board
    board = np.empty((5, 5), dtype=object)
    
    for _, row in board_df.iterrows():
        
        # Udtræk position fra tile_filnavn (fx tile_2_3.jpg → (2,3))
        row_idx, col_idx = parse_tile_position(row["tile_file"])
        
        # Gem både terrain + crowns direkte fra CSV
        board[row_idx,col_idx] = {
            "terrain": row["label"],
            "crowns": row["crowns"]
        }
    return board


# ------------------------------------------------------------
# 3. Breadth First Search (BFS) algoritmen til at find regioner
# ------------------------------------------------------------

def explore_region(board, start_row, start_col, visited, board_name, board_crowns):
    
    """ 
    Finder en sammmenhængende region og returnere 
    dens anatl score. Dvs. region_size * crowns !
    """
    
    # Hente start række og klonnen fra board
    tile = board[start_row, start_col]
    
    if tile is None:     # Returner 0, hvis ingen score
        return 0
    
    # Henter trræntypen 
    terrain_type = tile["terrain"]
    
    if terrain_type == "Empty":   # Inogerer hvis terræn er tom
        return 0
    
    queue = deque()   # Oprettes en kø til BFS
    
    # indsætter start-cellen i køen, så vi kan begynde at udforske regionen
    queue.append((start_row, start_col))    
     
    region_size = 0
    total_crowns = 0
    
    # kører så længe der er element i kø 
    while queue:
        
        # Fjerner det første element i køen, vi tager --> næste cell der skal undersøges
        row, col = queue.popleft()
        
        
        # Spring over hvis cellen er allered besøgt
        if visited[row, col]:
            continue
        
        # Marker cellen som besøget
        visited[row, col] = True
        
        current_tile = board[row, col] # Henter den aktuelle tile
        
        region_size+= 1
        
        # Lægger crowns fra denne tile til totalen
        total_crowns += current_tile["crowns"]
        
        
         # Definerer nabo-retninger (op, ned, venstre, højre)
        nieghbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        
        for dr, dc in nieghbors:  
            
            # Beregn nye position
            new_row = row + dr 
            new_col = col + dc
            
             # Tjekker at vi stadig er inden for boardet (5x5)
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                
                # Tjek at naboen ikke allered besøgt
                if not visited[new_row, new_col]: 
                    
                    # Hente naboe-tile
                    nieghbor_tile = board[new_row, new_col]
                    
                    # Tjekker at naboen findes 
                    if nieghbor_tile is not None:
                        
                        # Hvis naboen har samme terræn → tilføj til region
                        if nieghbor_tile["terrain"] == terrain_type:
                            queue.append((new_row, new_col))
                        
    return region_size * total_crowns           
    
# -----------------------------
# 4. Beregn score for et board
# -----------------------------    

def compute_board_score(board, board_name, board_crowns):
    
    """ 
    Summerer score for hele boardet!
    """
    
    # Opretter en 5x5 matrix til at holde styr på hvilke tiles der er besøgt (False = ikke besøgt)
    visited = np.zeros((5, 5), dtype=bool)
    total_score = 0
    
    for row in range(5):
        for col in range(5):
            
            # Hvis denne celle endnu ikke er besøgt
            if not visited[row, col]:
                
             # Tjekker at der faktisk findes en tile (ikke None)
                if board[row, col] is not None:
                    
                    # Finder score for hele regionen (via BFS)
                    region_score = explore_region(
                        board, row, col, visited, board_name, board_crowns
                    )
                    total_score += region_score

    return total_score

# -------------------------------
# 5. Beregn score for alle borads
# -------------------------------

def compute_score_from_csv(csv_path):
    """ 
    Input: CSV med terrain + crowns → Output: score per board
    """
    # Indlæser CSV-filen features_with_crowns
    df = pd.read_csv(csv_path)
    
    scores = {}
    
    # Grupperer alle tiles efter board_name (så vi behandler et board ad gangen)
    grouped = df.groupby("board_name")
    
    for board_name, board_df in grouped:
    
        # Konverterer dataframe for dette board til en 5x5 matrix
        board_matrix = build_board_matrix(board_df)
        
        # Beregner score for boardet (sender tom dict for crowns)
        score = compute_board_score(board_matrix, board_name, None)
        
        # Gemmer resultatet i dictionary
        scores[board_name] = score
        
    return scores