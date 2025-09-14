"""
Skripta za prikupljanje podataka i treniranje modela prepoznavanja gesta ruke.

Funkcionalnosti:
1. Prikupljanje podataka pomoću MediaPipe Hands:
   - Snima 21 landmark (x,y,z) = 63 vrijednosti po ruci
   - Sprema i informaciju o "handedness" (lijeva/desna ruka)
   - Pohranjuje podatke u JSON datoteke u `data/raw/`

2. Treniranje modela (MLP u TensorFlow/Keras):
   - Učitava JSON uzorke
   - Normalizira landmarkove (x,y) po bounding boxu ruke
   - Dijeli podatke na train/val skup
   - Gradi i trenira model
   - Sprema model i oznake u `models/`
"""

import os
import cv2
import time
import uuid
import json
import glob
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split

# -------------------- KONFIGURACIJA --------------------
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

# Parametri prikupljanja
SAMPLES_PER_CLASS = 80   # broj uzoraka po klasi
SAVE_PAUSE_SEC = 0.1     # pauza između spremanja uzoraka

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -------------------- FUNKCIJE --------------------
def preprocess_landmarks_xy(landmarks, handedness=None):
    """
    Normalizacija (x,y) koordinata po bounding boxu ruke.
    """
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


def collect_data():
    """
    Pokreće prikupljanje podataka s kamere.
    Tipke:
      - 0..9: odabir trenutne labele
      - s: spremi uzorak
      - q: prekid
    """
    print("Oznake:", GESTURE_LABELS)
    print("Pritisni broj 0-{} za labelu, 's' za spremanje uzorka, 'q' za izlaz".format(len(GESTURE_LABELS)-1))

    current_label_idx = None
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera nije dostupna")

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:
        saved_counts = {label: 0 for label in GESTURE_LABELS}
        last_save_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label
                # Zrcaljenje handedness zbog kamere
                if handedness == "Right":
                    handedness = "Left"
                elif handedness == "Left":
                    handedness = "Right"

                # Crtanje landmarkova
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Ekstrakcija koordinata
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                # UI info
                cv2.putText(frame_bgr, f"Label: {GESTURE_LABELS[current_label_idx] if current_label_idx is not None else '-'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame_bgr, f"Saved: {saved_counts}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            else:
                coords = None
                handedness = None
                cv2.putText(frame_bgr, "Nema ruke", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Prikupljanje podataka", frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            # Odabir labele tipkama 0..9
            if ord('0') <= key <= ord('9'):
                idx = key - ord('0')
                if idx < len(GESTURE_LABELS):
                    current_label_idx = idx
            elif key == ord('q'):
                break
            elif key == ord('s') and coords is not None and current_label_idx is not None:
                now = time.time()
                if now - last_save_time < SAVE_PAUSE_SEC:
                    continue
                last_save_time = now
                label = GESTURE_LABELS[current_label_idx]
                sample = {
                    "label": label,
                    "handedness": handedness,
                    "landmarks": coords,
                }
                fname = os.path.join(RAW_DIR, f"{int(now*1000)}_{uuid.uuid4().hex}.json")
                with open(fname, "w") as f:
                    json.dump(sample, f)
                saved_counts[label] += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Gotovo prikupljanje.")


def train_model():
    """
    Učitava JSON uzorke i trenira MLP model.
    """
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    print(f"Broj datoteka: {len(files)}")

    if len(files) == 0:
        print("Nema prikupljenih podataka. Pokreni prikupljanje.")
        return

    X, y = [], []
    for fp in files:
        with open(fp, "r") as f:
            sample = json.load(f)
        raw_label = sample["label"]
        mapped_label = TF_LABEL_REMAP.get(raw_label, raw_label)
        if mapped_label not in TF_LABELS:
            continue
        coords = sample["landmarks"]
        handedness = sample.get("handedness")

        # Pretvori JSON podatke u listu (x,y) točaka
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
        return

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # Definicija modela
    model_tf = tf.keras.models.Sequential([
        tf.keras.layers.Input((21*2,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model_tf.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
    ]

    history = model_tf.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) else None,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Spremanje modela
    model_tf.save(TF_MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"labels": TF_LABELS}, f)
    print("TF model spremljen u:", TF_MODEL_PATH)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    print("Odaberi opciju:")
    print("1 - Prikupljanje podataka")
    print("2 - Treniranje modela")
    choice = input("Unesi broj: ")
    if choice == "1":
        collect_data()
    elif choice == "2":
        train_model()
    else:
        print("Nepoznata opcija.")
