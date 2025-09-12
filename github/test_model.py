"""
Skripta za testiranje/inferenciju treniranog modela gesta ruke.

- Učitava spremljeni TensorFlow model i mapu oznaka iz `models/`
- Omogućuje real-time inferenciju s web kamere koristeći MediaPipe Hands
- Sadrži brzu provjeru na jednom uzorku iz `data/raw`
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

# Pretprocesiranje landmarkova (ista logika kao u treniranju): normalizacija (x,y)
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


# Učitaj spremljeni TF model i mapu oznaka ako postoje
if os.path.exists(TF_MODEL_PATH):
    model_tf = tf.keras.models.load_model(TF_MODEL_PATH)
    with open(META_PATH, "r") as f:
        LABELS = json.load(f)["labels"]
    print("Model učitan:", TF_MODEL_PATH)
else:
    model_tf = None
    LABELS = TF_LABELS
    print("Nema spremljenog modela. Pokreni train_model.py.")

# --- Real-time kamera inferencija ---
def run_realtime_inference():
    # Otvori default kameru
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera nije dostupna")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Postavi MediaPipe Hands u stream modu (praćenje)
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:
        while True:
            # Čitanje frame-a iz kamere
            ok, frame = cap.read()
            if not ok:
                continue
            # Prebaci u RGB za MediaPipe i procesiraj ruku
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            pred_label, pred_prob = None, None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Nacrtaj landmarkove i veze na frame-u
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                handed = None
                if results.multi_handedness:
                    handed = results.multi_handedness[0].classification[0].label

                # Ekstraktiraj značajke (normalizirani x,y) za model
                feats_xy = preprocess_landmarks_xy(hand_landmarks.landmark, handedness=handed).reshape(1, -1)

                if model_tf is not None:
                    # Predikcija vjerojatnosti po klasama i odabir najvjerojatnije
                    probs = model_tf.predict(feats_xy, verbose=0)[0]
                    idx = int(np.argmax(probs))
                    pred_label = LABELS[idx]
                    pred_prob = float(np.max(probs))

            # HUD: status modela i rezultat predikcije
            cv2.putText(frame_bgr, f"Model: {'TF' if model_tf else 'Nije ucitan'}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if pred_label is not None and pred_prob >= 0.6:
                cv2.putText(frame_bgr, f"Predikcija: {pred_label} ({pred_prob:.2f})", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            elif pred_label is not None:
                cv2.putText(frame_bgr, f"Nepouzdano: {pred_label} ({pred_prob:.2f})", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Inferencija (TF)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# --- Brza provjera na jednom uzorku iz `data/raw` ---
def quick_check():
    # Nađe prvu dostupnu JSON datoteku i proslijedi je kroz model
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        print("Nema podataka za provjeru.")
        return

    with open(files[0], "r") as f:
        s = json.load(f)

    coords = s["landmarks"]
    # Ovdje se uzima sirovih 42 vrijednosti (x,y) bez normalizacije
    # Napomena: Za konzistentnost s treniranjem bolje je koristiti istu normalizaciju
    xy = []
    for i in range(21):
        xy.append(coords[3*i])
        xy.append(coords[3*i+1])
    x_tf = np.array(xy, dtype=np.float32).reshape(1, -1)

    if model_tf is not None:
        probs = model_tf.predict(x_tf, verbose=0)[0]
        idx = int(np.argmax(probs))
        print("TF predikcija:", LABELS[idx], float(np.max(probs)))
    else:
        print("TF model nije ucitan.")

if __name__ == "__main__":
    # Pokreni ili quick_check() ili run_realtime_inference()
    quick_check()
    # run_realtime_inference()


