import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# -------------------------
# 1. Define test boards
# -------------------------
TEST_BOARDS = {
    "board_1","board_5","board_19","board_23",
    "board_25","board_29","board_35","board_39",
    "board_49","board_53","board_67","board_70"
}

# -------------------------
# 2. Load data
# -------------------------
df = pd.read_csv("features_with_crowns.csv")

# -------------------------
# 3. Split data
# -------------------------
train_df = df[~df["board_name"].isin(TEST_BOARDS)]
test_df  = df[df["board_name"].isin(TEST_BOARDS)]

# -------------------------
# 4. Features / labels
# -------------------------
X_train = train_df.drop(columns=["board_name","tile_file","label","crowns"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["board_name","tile_file","label","crowns"])
y_test = test_df["label"]

# -------------------------
# 5. Scaling
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -------------------------
# 6. Train final model
# (med dine bedste parametre)
# -------------------------
model = SVC(kernel="rbf", C=10, gamma="scale")
model.fit(X_train, y_train)

# -------------------------
# 7. Predict
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# 8. Evaluation
# -------------------------
print("\n=== Test Accuracy ===")
print(accuracy_score(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# -------------------------
# 9. Plot confusion matrix
# -------------------------

labels = sorted(y_test.unique())  # dine terrain classes

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)

plt.title("Confusion Matrix - Final Test Set")
plt.show()