
""" Denne fil kombinerer to CSV/datasæt:
- evaluation_per_tile.csv (detekterede crowns pr. tile)
- features.csv (HSV-features og terrain labels)
Output er en samlet CSV med både features, labels og crowns pr. tile."""

import pandas as pd

# Load de 2 CSV'er
df_features = pd.read_csv("features.csv")
df_crowns = pd.read_csv("evaluation_tiles.csv", sep=";")


# Gør klonnenavns ens
df_crowns = df_crowns.rename(columns={
    "Board": "board_name",
    "Tile": "tile_file",
    "Detected_Crowns": "crowns"
})

# Tilføjer jpg. så det matcher
df_crowns["tile_file"] = df_crowns["tile_file"] + ".jpg"

# Merge på board + tile
df_merged = df_features.merge(
    df_crowns[["board_name", "tile_file", "crowns"]],
    on=["board_name", "tile_file"],
    how="left"
)

# hvis nogen mangler --> sæt 0
df_merged["crowns"] = df_merged["crowns"].fillna(0)

# Ændr rækkefølgen af kolonner
feature_cols = [col for col in df_merged.columns if col.startswith(("hue_", "sat_", "val_"))]

# Resten i den rækkefølge 
ordered_cols = feature_cols + ["board_name", "tile_file", "label", "crowns"]
df_merged = df_merged[ordered_cols]

# Gem nye CSV
df_merged.to_csv("features_with_crowns.csv", index=False)
print("Done! new CSV med crowns")