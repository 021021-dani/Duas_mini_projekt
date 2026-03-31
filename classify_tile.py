"""
Modul: Terrænklassificering ved hjælp af HSV-histogrammer og scikit-learns KNN.
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from feature_extraction import extrac_hsv_histogram


class TileClassifier:
    """Klassificerer terrænet for en enkelt tile."""

    def __init__(self, features_csv="features.csv", k=3):
        self.k = k
        self.features_csv = Path(features_csv)

        # Initialiser KNN-modellen med euklidisk afstand og ensartet vægtning
        self.knn_model = KNeighborsClassifier(
            n_neighbors=self.k, metric="euclidean", weights="uniform"
        )
        self.is_fitted = False
        self._load_and_train_model()

    def _load_and_train_model(self):
        """
        Læser features.csv og træner scikit-learn KNN-modellen.
        """

        # Afbryd hvis træningsdata ikke findes
        if not self.features_csv.exists():
            print(f"Warning: Training data {self.features_csv} not found.")
            return

        training_data = []  # Samler alle features (X)
        labels = []  # Samler alle labels (y)

        # Åbn CSV-filen og læs data
        with self.features_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # Spring over header-rækken (kolonnenavne)

            # Gennemgå hver række i CSV-filen
            for row in reader:
                # Features er de første 20 værdier (indeks 0-19), label er på indeks 20
                if len(row) >= 22 or (
                    len(row) == 22 and "Unknown" not in row
                ):  # Tilpas dette hvis CSV-formatet ændrer sig
                    # Konverter de første 20 celler til et float-array
                    features = np.array(row[:20], dtype=float)
                    label = row[20]

                    training_data.append(features)
                    labels.append(label)

        # Træn kun modellen, hvis der rent faktisk er indlæst data
        if training_data:
            # Konverter listerne til numpy arrays for scikit-learn
            X = np.array(training_data)
            y = np.array(labels)

            # Træn scikit-learn KNN-modellen
            self.knn_model.fit(X, y)
            self.is_fitted = True  # Marker at modellen nu er klar til brug

    def classify(self, cell_image: np.ndarray) -> str:
        """
        Klassificerer en enkelt tile ved hjælp af scikit-learn KNN.
        Returnerer en tekststreng med terrænets navn.
        """

        # Hvis modellen ikke er trænet, kan vi ikke klassificere
        if not self.is_fitted:
            return "Unknown"

        # Udtræk 20 HSV features fra tilen
        features = extrac_hsv_histogram(cell_image)

        # KNeighborsClassifier forventer et 2D array, så vi reshaper
        # fra (20,) til (1, 20)
        features_2d = features.reshape(1, -1)

        # Forudsig terrænet ved hjælp af den trænede model
        prediction = self.knn_model.predict(features_2d)

        return prediction[0]
