
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("features_with_crowns.csv")

# Features (20 HS
# V værdier)
X = df.drop(columns=["board_name", "tile_file", "label", "crowns"])

# Labels (terrain)
y = df["label"]

# PCA til 3D
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Farver til hver terrain (HER skal den stå – før loop)
color_map = {
    "Water": "blue",
    "Field": "gold",
    "Grass": "cyan",
    "Forest": "green",
    "Swamp": "gray",
    "Mine": "black",
    "Home": "pink",
    "Empty": "brown"
}

# Plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot hver klasse separat
for terrain in sorted(y.unique()):
    idx = y == terrain
    
    ax.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        X_pca[idx, 2],
        label=terrain,
        color=color_map.get(terrain, "black"),  # bruger dine farver
        s=20,
        alpha=0.7
    )

# Labels og titel
ax.set_title("3D PCA Visualization of Terrain Features")
ax.set_xlabel("1st Egigenvector")
ax.set_ylabel("2nd Egigenvector")
ax.set_zlabel("3rd Egigenvector")

# Bedre vinkel (valgfrit men anbefalet)
ax.view_init(elev=20, azim=45)

ax.legend()
plt.show()