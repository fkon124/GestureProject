"""
Skripta za treniranje modela prepoznavanja gesta ruke na temelju 2D landmarkova.

- Učitava anotirane uzorke iz `data/raw/*.json`
- Normalizira landmarkove (x,y) po bounding boxu ruke
- Dijeli podatke na train/val skup
- Gradi i trenira jednostavnu MLP (Dense) mrežu u TensorFlow/Keras
- Sprema trenirani model i mapu oznaka u `models/`
"""

import os
import json
import glob
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Popis gesta (oznake klasa) i osnovne staze/datoteke
GESTURE_LABELS = [
    "OK","Finger gun","Ljubav","Poziv","Thumbs Up","Thumbs Down",
    "Peace","Point","Fist","Open Hand"
]
TF_LABELS = GESTURE_LABELS
NUM_CLASSES = len(TF_LABELS)

TF_LABEL_REMAP = {label: label for label in TF_LABELS}

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
MODEL_DIR = "models"
TF_MODEL_PATH = os.path.join(MODEL_DIR, "hand_gesture_tf.keras")
META_PATH = os.path.join(MODEL_DIR, "labels.json")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Pretprocesiranje landmarkova: normalizacija (x,y) koordinata po lokalnom okviru ruke
# - Iz ulaznih landmark objekata uzima x i y
# - Računa min/max po x i y kako bi dobio širinu/visinu okvira
# - Svaku točku skalira na [0,1] unutar tog okvira (invarijantno na skalu i pomak)
def preprocess_landmarks_xy(landmarks, handedness=None):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)
    w, h = max_x - min_x, max_y - min_y
    if w == 0: w = 1e-6
    if h == 0: h = 1e-6
    normalized = []
    for lm in landmarks:
        nx = (lm.x - min_x) / w
        ny = (lm.y - min_y) / h
        normalized.extend([nx, ny])
    return np.array(normalized, dtype=np.float32)


# Učitavanje skupa podataka iz JSON datoteka (svaka sadrži: label, landmarks, handedness)
files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
print(f"Broj datoteka: {len(files)}")

if len(files) == 0:
    print("Nema prikupljenih podataka. Pokreni prikupljanje.")
else:
    X, y = [], []
    for fp in files:
        # Učitaj jedan uzorak (jedna slika/snimka ruke s anotacijom)
        with open(fp, "r") as f:
            sample = json.load(f)
        raw_label = sample["label"]
        mapped_label = TF_LABEL_REMAP.get(raw_label, raw_label)
        if mapped_label not in TF_LABELS:
            continue
        coords = sample["landmarks"]
        handedness = sample.get("handedness")

        # Iz JSON-a dolazi niz od 63 vrijednosti (21 točaka x,y,z). Ovdje koristimo samo x,y.
        # Sintetiziramo minimalan objekt s atributima x,y za svaku od 21 točke.
        xy = preprocess_landmarks_xy(
            [type("L",(object,),{"x":coords[3*i],"y":coords[3*i+1]}) for i in range(21)],
            handedness=handedness
        )
        X.append(xy)
        y.append(TF_LABELS.index(mapped_label))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print("X shape:", X.shape, "y shape:", y.shape)

    if len(X) == 0:
        print("Nema valjanih uzoraka nakon remapiranja.")
    else:
        # Stratificirana podjela na trening i validaciju (ako postoji više od jedne klase)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None
        )

        # Definicija jednostavnog MLP modela za klasifikaciju gesta
        model_tf = tf.keras.models.Sequential([
            tf.keras.layers.Input((21*2,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        ])

        # Kompilacija modela: Adam optimizator, SparseCategoricalCrossentropy za integer oznake
        model_tf.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        # Rani prekid na temelju val točnosti, vraća najbolje težine
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        ]

        # Treniranje modela
        history = model_tf.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if len(X_val) else None,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Spremi model i pripadni popis oznaka
        model_tf.save(TF_MODEL_PATH)
        with open(META_PATH, "w") as f:
            json.dump({"labels": TF_LABELS}, f)
        print("TF model spremljen u:", TF_MODEL_PATH)
