import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# -------------------------
# Define test boards
# -------------------------
TEST_BOARDS = {
    "board_1","board_5","board_19","board_23",
    "board_25","board_29","board_35","board_39",
    "board_49","board_53","board_67","board_70"
}


def test_svm_classifier(features_csv="features_with_crowns.csv"):
    """
    Train and evaluate SVM classifier on terrain data.
    Loads features, splits into train/test, trains SVM, and displays metrics.
    """
    # -------------------------
    # 1. Load data
    # -------------------------
    df = pd.read_csv(features_csv)

    # -------------------------
    # 2. Split data
    # -------------------------
    train_df = df[~df["board_name"].isin(TEST_BOARDS)]
    test_df  = df[df["board_name"].isin(TEST_BOARDS)]

    # -------------------------
    # 3. Features / labels
    # -------------------------
    X_train = train_df.drop(columns=["board_name","tile_file","label","crowns"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["board_name","tile_file","label","crowns"])
    y_test = test_df["label"]

    # -------------------------
    # 4. Scaling
    # -------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # -------------------------
    # 5. Train final model
    # -------------------------
    model = SVC(kernel="rbf", C=10, gamma="scale")
    model.fit(X_train, y_train)

    # -------------------------
    # 6. Predict
    # -------------------------
    y_pred = model.predict(X_test)

    # -------------------------
    # 7. Evaluation
    # -------------------------
    print("\n=== Test Accuracy ===")
    print(accuracy_score(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # -------------------------
    # 8. Plot confusion matrix
    # -------------------------
    labels = sorted(y_test.unique())  # dine terrain classes

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)

    plt.title("Confusion Matrix - Final Test Set")
    plt.show()
    
    return model, scaler


if __name__ == "__main__":
    test_svm_classifier()