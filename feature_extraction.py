import cv2 as cv
import numpy as np
import os
import csv
import matplotlib.pyplot as plt


def extrac_hsv_histogram(tile):

    """
    Udtræk farvehistogram fra en tile i HSV-farverum.
    Returnerer en numpy array med 20 værdier.   
    """
    
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)  # Konverter fra RGB til HSV

    # Beregn histogram for hver kanal ([0] = kanal, None = ingen maske, 
    # [10] = antal bins, [0,180] = værdiområde for HSV
    
    h_hist = cv.calcHist([hsv], [0], None, [10], [0, 180])
    s_hist = cv.calcHist([hsv], [1], None, [5],  [0, 256])
    v_hist = cv.calcHist([hsv], [2], None, [5],  [0, 256])

    # Normaliser så lysstryke ikke dominerer- alle værdier mellem 0 og 1
    
    h_hist = cv.normalize(h_hist, h_hist).flatten()
    s_hist = cv.normalize(s_hist, s_hist).flatten()
    v_hist = cv.normalize(v_hist, v_hist).flatten()

    # Sæt de tre histogrammer sammen til én vektor med 20 tal
    features = np.concatenate([h_hist, s_hist, v_hist])

    return features

def process_all_tiles(tiles_root_folder, output_csv):
    
    """
    Gennemgår alle board-mapper og deres tiles,
    udtrækker features og gemmer i en CSV-fil.
    """

    # Klonner til CSV-filen (10 hue, 5 sat, 5 val)
      
    header = (
        [f"hue_{i}" for i in range(10)] +
        [f"sat_{i}" for i in range(5)]  +
        [f"val_{i}" for i in range(5)]  +
        ["label", "tile_file"]
    )
    
    
    rows = []  # Samler alle rækker inden de skrives til CSV 

    # Gå gennem alle borad-mapper, sikrer med sorted  at de kommer i alfabetisk rækkefølge
    
    for board_name in sorted(os.listdir(tiles_root_folder)):
        board_path = os.path.join(tiles_root_folder, board_name)
        
        # Spring over hvis det ikke er en mappe
        
        if not os.path.isdir(board_path):
            continue
        print(f"Treats: {board_name}")

        # Gå gennem alle tiles-billeder inde i board-mappen
        
        for tile_file in sorted(os.listdir(board_path)):
            if not tile_file.lower().endswith(".jpg"):
                continue

            # Byg den fulde sti til tile-billedet
            
            tile_path = os.path.join(board_path, tile_file)
            
            tile = cv.imread(tile_path) # Indlæse billedet med cv2
            
            # Spring over hvis billedet ikke kunne læses
            
            if tile is None:
                print(f"  Can not read:{tile_file}")
                continue

            # Udtræk de 20 histogram-værdier fra tilen
            features = extrac_hsv_histogram(tile)
            
            # Bygge en række til CSV, runder 4 decmimaler + label og filnav
            
            row = list(np.round(features, 4)) + ["Unknow", tile_file]
            rows.append(row)
            
    # Skrives alle rækker til CSV-filen på en gang
    # Newline == er nødvendige på Windows så der ikke kommer tomme linjer 
    
    with open(output_csv, "w", newline="") as f: #  f variablen bruge  i stedet for at skrive hele filnavnet igen.
        
        # csv.writer(f) opretter et writer-objekt der ved hvordan man skriver til en CSV-fil,
        # det håndterer automatisk kommaer og anførselstegn korrekt.
        writer = csv.writer(f) 

        writer.writerow(header)  # Første linje = klonnenavne 
        writer.writerows(rows)  # Alle efterfølgende linjer = data 

    print(f"\nFinished! {len(rows)} tiles save in: {output_csv}")
    
    

def draw_histogram(ax, hist, colors, title, labels):
    """Tegner ét histogram med markering af den højeste søjle"""
    
    # Tegn alle søjler
    ax.bar(range(len(hist)), hist, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(title)
    ax.set_xticks(range(len(hist)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Normaliseret frekvens")

    # Find og marker den højeste søjle med fed kant og værdi
    max_idx = np.argmax(hist)
    ax.bar(max_idx, hist[max_idx], color=colors[max_idx], edgecolor='black', linewidth=1.5)
    ax.text(max_idx, hist[max_idx] + 0.02, f'{hist[max_idx]:.2f}',
            ha='center', fontsize=9, fontweight='bold')


def visualize_tile_and_histogram(tile_path):
    """
    Viser tile-billedet og dets HSV-histogram side om side
    så man kan se sammenhængen mellem billede og tal.
    """

    # Indlæs billedet fra stien
    tile = cv.imread(tile_path)

    # Stop hvis billedet ikke kunne læses
    if tile is None:
        print("Can not read:", tile_path)
        return

    # Konverter fra BGR til HSV for farveanalyse
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)

    # Konverter til RGB så matplotlib kan vise billedet korrekt
    tile_rgb = cv.cvtColor(tile, cv.COLOR_BGR2RGB)

    # Beregn og normaliser histogrammerne så værdier ligger mellem 0 og 1
    h_hist = cv.normalize(cv.calcHist([hsv], [0], None, [10], [0, 180]), None).flatten()
    s_hist = cv.normalize(cv.calcHist([hsv], [1], None, [5],  [0, 256]), None).flatten()
    v_hist = cv.normalize(cv.calcHist([hsv], [2], None, [5],  [0, 256]), None).flatten()

    # Farver til søjlerne - følger farvehjulet fra rød til pink
    hue_colors = ['#FF4444','#FF8800','#AACC00','#44BB00','#00AA88',
                  '#0088FF','#4444FF','#8800FF','#CC0088','#FF4444']

    # Farver til saturation - fra grå til stærk blå
    sat_colors = ['#BBBBBB','#8899BB','#4477BB','#1144AA','#001188']

    # Farver til value - fra sort til hvid
    val_colors = ['#111111','#555555','#999999','#CCCCCC','#FFFFFF']

    # Labels til x-akserne
    hue_labels = ['rød','org','gul\ngrøn','grøn','cyan','blå','mørk\nblå','lilla','pink','rød']
    sat_labels = ['grå\nmat','lav','middel','høj','stærk\nfarve']
    val_labels = ['meget\nmørk','mørk','middel','lys','meget\nlys']

    # Lav et vindue med 4 plots side om side - billede + 3 histogrammer
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Sæt filnavnet som titel øverst på vinduet
    fig.suptitle(os.path.basename(tile_path), fontsize=13)

    # Vis selve tile-billedet i første plot
    axes[0].imshow(tile_rgb)
    axes[0].set_title("Tile")
    axes[0].axis('off')

    # Tegn de tre histogrammer med hjælpefunktionen
    draw_histogram(axes[1], h_hist, hue_colors, "Hue (farvetone)",    hue_labels)
    draw_histogram(axes[2], s_hist, sat_colors, "Saturation (mætning)", sat_labels)
    draw_histogram(axes[3], v_hist, val_colors, "Value (lysstyrke)",  val_labels)

    plt.tight_layout()
    plt.show()