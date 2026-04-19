"""
Responsibility: Execute Kingdomino rules on the 5x5 data matrix.
"""

"""Ansvar:
# - Opbygge 5x5 array(board) fra CSV
# - Anvendes BFS til at finde regioner
# - Beregne Kingdomino score(antal Kongekroner)+ antal sammenhængende områder"""

import pandas as pd
import numpy as np
import cv2 as cv

# Bruges til at oprette tupler/lister, hvor man effektivt kan tilføje og fjerne elementer
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


# ------------------------
# 2. Load ground truth 
# ------------------------

def load_board_crowns(csv_path):
    
    """ 
    Loader PoinScore_ground_truth_csv. og laver:
    borad_name --> terræn ---> Kroner mapping!
    """

    df = pd.read_csv(csv_path, sep=";")
    df = df.fillna(0)
    
    board_crowns = {}
    
    for _, row in df.iterrows():
        board_name = row["board_name"]
        
        board_crowns[board_name] = {
            
            "Water" : row["Water"],
            "Field" : row["Field"],
            "Grass" : row["Grass"],
            "Forest" : row["Forest"],
            "Mine" : row["Mine"],
            "Swamp" : row["Swamp"],
            "Home" : row["Home"],
            "Empty" : row["Empty"]       
        }
        
    return board_crowns

# ----------------------
# 3. Byg board (5 x 5)
# ---------------------

def build_board_matrix(board_df):
    
    """ 
    Konverter Dataframe til ---> 5 x 5 numpy array,
    Hver Celle: {"terrain": str, "crowns": int}, 
    Crowns beregnes senere fra GT !
    """
    
    board = np.empty((5, 5), dtype=object)
    
    for _, row in board_df.iterrows():
        row_idx, col_idx = parse_tile_position(row["tile_file"])
        
        board[row_idx,col_idx] = {
            "terrain": row["label"],
            "crowns": row.get("crowns", 0)
        }
    return board


# ------------------------------------------------------------
# 4. Breadth First Search (BFS) algoritmen til at find regioner
# ------------------------------------------------------------

def explore_region(board, start_row, start_col, visited, board_name, board_crowns):
    
    """ 
    Finder en sammmenhængende region og returnere 
    dens anatl score. Dvs. region_size * crowns !
    """
    
    tile = board[start_row, start_col]
    
    if tile is None:
        return 0
    
    terrain_type = tile["terrain"]
    
    if terrain_type == "Empty":   # Inogerer hvis terræn er tom
        return 0
    
    queue = deque()   # Oprettes en kø til BFS
    
    # indsætter start-cellen i køen, så vi kan begynde at udforske regionen
    queue.append((start_row, start_col))    
     
    region_size = 0
    total_crowns = 0
    
    while queue:
        
        # Fjerner og returnerer det første element i køen, vi tager 
        #  den næste celle (row, col), som skal udforskes i BFS
        
        row, col = queue.popleft()
        
        if visited[row, col]:
            continue
        
        visited[row, col] = True
        
        current_tile = board[row, col]
        
        region_size+= 1
        total_crowns += current_tile["crowns"]
        
        
        # 4 retninger ---> op, ned, vestre, højre
        nieghbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in nieghbors:  
            
            new_row = row + dr 
            new_col = col + dc
            
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                
                if not visited[new_row, new_col]:  
                    nieghbor_tile = board[new_row, new_col]
                    
                    if nieghbor_tile is not None:
                        if nieghbor_tile["terrain"] == terrain_type:
                            queue.append((new_row, new_col))
                        
    return region_size * total_crowns           
    
# -----------------------------
# 5. Beregn score for et board
# -----------------------------    

def compute_board_score(board, board_name, board_crowns):
    
    """ 
    Summerer score for hele boardet!
    """
    
    visited = np.zeros((5, 5), dtype=bool)
    total_score = 0
    
    for row in range(5):
        for col in range(5):
            
            if not visited[row, col]:
                if board[row, col] is not None:
                    
                    region_score = explore_region(
                        board, row, col, visited, board_name, board_crowns
                    )

                    total_score += region_score

    return total_score

# -------------------------------
# 6. Beregn score for alle borads
# -------------------------------

def compute_score_from_csv(csv_path, crown_csv_path):
    """ 
    Input: feature_csv + Crowns CSV--> Output: Dictionary med score per board!
    """
    
    df = pd.read_csv(csv_path)
    
    # GT PointScore loades her
    board_crowns = load_board_crowns(crown_csv_path)
    
    scores = {}
    
    grouped = df.groupby("board_name")
    
    for board_name, board_df in grouped:
        board_matrix = build_board_matrix(board_df)
        
        score = compute_board_score(board_matrix, board_name, board_crowns)
        scores[board_name] = score
        
    return scores

        
    

            

        