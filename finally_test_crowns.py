import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------
# 1. Load data
# -------------------------
df = pd.read_csv("evaluation_tiles.csv", sep=";")

# -------------------------
# 2. Prepare and clean data
# -------------------------
df["Is_Test_Set"] = df["Is_Test_Set"].astype(str).str.lower() == "true"
df["GT_Crowns"] = pd.to_numeric(df["GT_Crowns"], errors="coerce")
df["Detected_Crowns"] = pd.to_numeric(df["Detected_Crowns"], errors="coerce")
df["Error"] = pd.to_numeric(df["Error"], errors="coerce")

df = df.dropna(subset=["GT_Crowns", "Detected_Crowns", "Error"]).copy()

df["GT_Crowns"] = df["GT_Crowns"].astype(int)
df["Detected_Crowns"] = df["Detected_Crowns"].astype(int)
df["Error"] = df["Error"].astype(int)

# -------------------------
# 3. Split data
# -------------------------
train_df = df[~df["Is_Test_Set"]]
test_df = df[df["Is_Test_Set"]]

# -------------------------
# 4. Ground truth / predictions
# -------------------------
y_train_true = train_df["GT_Crowns"]
y_train_pred = train_df["Detected_Crowns"]

y_test_true = test_df["GT_Crowns"]
y_test_pred = test_df["Detected_Crowns"]

# -------------------------
# 5. Tile-level crown metrics
# -------------------------
test_mae = mean_absolute_error(y_test_true, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
test_bias = (y_test_pred - y_test_true).mean()  # signed average error
test_exact_match = (y_test_pred == y_test_true).mean()

# -------------------------
# 6. Print summary
# -------------------------
print("\n=== Crown Detection (Tile-Level, Test Set) ===")
print(f"Tiles in test set: {len(test_df)}")
print(f"MAE: {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"Mean signed error (bias): {test_bias:.4f}")
print(f"Exact match ratio: {test_exact_match:.4f}")

print("\n=== Signed Error Distribution (Test Set) ===")
error_dist = test_df["Error"].value_counts().sort_index()
print(error_dist)

print("\n=== Crown Totals (Test Set) ===")
print(f"GT total crowns: {int(y_test_true.sum())}")
print(f"Detected total crowns: {int(y_test_pred.sum())}")
print(f"Total crown difference: {int((y_test_pred - y_test_true).sum())}")

# -------------------------
# 7. MAE by board (test set)
# -------------------------
test_df = test_df.copy()
test_df["Absolute_Error"] = np.abs(test_df["Error"])

board_mae = test_df.groupby("Board", as_index=False)["Absolute_Error"].mean()
board_mae = board_mae.sort_values("Absolute_Error", ascending=False)

print("\n=== MAE by Board (Test Set) ===")
print(board_mae.to_string(index=False))

# -------------------------
# 8. Plot MAE evaluation
# -------------------------
plt.figure(figsize=(10, 4))
plt.bar(board_mae["Board"], board_mae["Absolute_Error"], color="steelblue")
plt.axhline(
    y=test_mae,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Overall test MAE = {test_mae:.4f}",
)
plt.title("MAE per Board - Crown Count per Tile (Test Set)")
plt.xlabel("Board")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# 9. Plot signed error distribution
# -------------------------
plt.figure(figsize=(6, 4))
error_dist.plot(kind="bar")
plt.title("Signed Error Distribution - Crown Count per Tile (Test Set)")
plt.xlabel("Error = Detected_Crowns - GT_Crowns")
plt.ylabel("Number of tiles")
plt.tight_layout()
plt.show()
