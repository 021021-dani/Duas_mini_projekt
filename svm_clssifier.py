
"""
Modul: Terrænklassificering ved hjælp af HSV-histogrammer og scikit-learns SVM.
"""


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit, cross_val_score
from sklearn.svm import SVC 
from feature_extraction import extrac_hsv_histogram

# Definer de spilleplader der sorteres fra til test-sættet
TEST_BOARDS = {
    "board_1",
    "board_5",
    "board_19",
    "board_23",
    "board_25",
    "board_29",
    "board_35",
    "board_39",
    "board_49",
    "board_53",
    "board_67",
    "board_70",
}

class TileClssifier:
    
    """Klassificerer terrænet for en enkelt tile."""
    
    def __init__(self, features_csv="features.csv", C = 1.0, kernel="rbf"):
        self.C = C
        self.kernel = kernel 
        self.features_csv = Path(features_csv)
        
        # Intialiserer SVM-model
        
        self.svm_model = SVC(C=self.C, kernel= self.kernel)
        self.is_fitted = False
        self._load_and_train_model()
        
    
    def _load_and_train_model(self):
        """
        Læser features.csv og træner scikit-learn SVM-modellen.
        """
        
        if not self.features_csv.exists():
            print(f"Warning: Training data {self.features_csv} not found.")
            return
        
        # Læs CSV-filen med pandas
        df = pd.read_csv(self.features_csv)
        
        # Træn SVM modellen, hvis der rent faktisk er indlæst data
        if not df.empty:
            df = df.copy()
            
            if "board_name" in df.columns:
                df = df[~df["board_name"].isin(TEST_BOARDS)].copy()
                
            X = df.iloc[:, :-3].to_numpy(dtype=float)
            y = df["label"].to_numpy(dtype=str)
            
            # Træn svm
            self.svm_model.fit(X, y)
            self.is_fitted = True
        
        def classify(self, cell_image: np.ndarray) -> str:
            
            """
            Klassificerer en enkelt tile ved hjælp af scikit-learn SVM.
            Returnerer en tekststreng med terrænets navn! 
            """
            
            if not self.is_fitted:
                return "Unknown"
            
            # Udtræk 20 HSV features fra tilen
            
            features = extrac_hsv_histogram(cell_image)
            features_2d = features.reshape(1, -1)
            
            prediction = self.svm_model.predict(features_2d)
            return prediction[0]
        
        
if __name__ == "__main__":

    # --- 1. Indlæs Dataset ---
    csv_path = "features.csv"
    print(f"\nBehandler data fra {csv_path}...")
    df = pd.read_csv(csv_path)

    fold_mapping = {
        
            # Fold 1
        "board_4": 0,
        "board_8": 0,
        "board_20": 0,
        "board_24": 0,
        "board_34": 0,
        "board_38": 0,
        "board_42": 0,
        "board_46": 0,
        "board_48": 0,
        "board_52": 0,
        "board_65": 0,
        "board_72": 0,
        # Fold 2
        "board_2": 1,
        "board_6": 1,
        "board_18": 1,
        "board_22": 1,
        "board_28": 1,
        "board_32": 1,
        "board_36": 1,
        "board_40": 1,
        "board_51": 1,
        "board_55": 1,
        "board_58": 1,
        "board_62": 1,
        # Fold 3
        "board_10": 2,
        "board_14": 2,
        "board_11": 2,
        "board_15": 2,
        "board_26": 2,
        "board_30": 2,
        "board_41": 2,
        "board_44": 2,
        "board_57": 2,
        "board_61": 2,
        "board_64": 2,
        "board_68": 2,
        
        # Fold 4
        "board_3": 3,
        "board_7": 3,
        "board_17": 3,
        "board_21": 3,
        "board_27": 3,
        "board_31": 3,
        "board_43": 3,
        "board_47": 3,
        "board_50": 3,
        "board_54": 3,
        "board_59": 3,
        "board_63": 3,
        
        # Fold 5
        "board_9": 4,
        "board_13": 4,
        "board_12": 4,
        "board_16": 4,
        "board_33": 4,
        "board_37": 4,
        "board_45": 4,
        "board_56": 4,
        "board_60": 4,
        "board_66": 4,
        "board_69": 4,
    }


    df = df[~df["board_name"].isin(TEST_BOARDS)].copy()
    df["fold"] = df["board_name"].map(fold_mapping)
    df = df.dropna(subset=["fold"]).copy()

    X_train_val = df.iloc[:, :20].to_numpy(dtype=float)
    y_train_val = df["label"].to_numpy(dtype=str)
    fold_indices = df["fold"].to_numpy(dtype=int)

    print(f"Total tiles til CV hyper-tuning: {len(X_train_val)} over 5 Folds.")

    ps = PredefinedSplit(test_fold=fold_indices)

    # --- 2. Hyperparameter tuning for SVM ---
    print("\nStarter parameter tuning for SVM...")

    svm = SVC()

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(svm, param_grid, cv=ps, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_val, y_train_val)

    best_params = grid.best_params_
    best_acc = grid.best_score_

    print(
        f"Bedste parametre: {best_params} med Validerings-accuracy på {(best_acc * 100):.2f}%\n"
    )

    # --- 3. Cross-validation ---
    print("Kører 5-Fold Cross-Validation med bedste SVM...")

    cv_scores = cross_val_score(
        grid.best_estimator_, X_train_val, y_train_val, cv=ps, scoring="accuracy"
    )


    for i, score in enumerate(cv_scores):
        print(f"  > Fold {i + 1} Accuracy: {(score * 100):.2f}%")

    print(f"\n  => Mean Accuracy: {(np.mean(cv_scores) * 100):.2f}%")
    print(f"  => Std: {(np.std(cv_scores) * 100):.2f}%")
            
            
        
                    
    
        
        
        
    
