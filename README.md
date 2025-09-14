# Hand Gesture Recognition

Projekt za prepoznavanje gesta ruke koristeći 2D landmarkove iz MediaPipe biblioteke i TensorFlow neuronskih mreža.

## Pregled

Ovaj sustav omogućuje:
- Prikupljanje podataka i treniranje modela za prepoznavanje 10 različitih gesta ruke
- Real-time inferenciju preko web kamere
- Normalizaciju landmark koordinata za robusnost prema skali i poziciji

### Podržani gesti

1. **OK** - Kružni znak s palcem i kažiprstom
2. **Finger gun** - Pokazivanje s kažiprstom
3. **Ljubav** - Srce oblikovano rukama
4. **Poziv** - Pozivni gest
5. **Thumbs Up** - Palac prema gore
6. **Thumbs Down** - Palac prema dolje  
7. **Peace** - Znak mira (V)
8. **Point** - Pokazivanje
9. **Fist** - Stisnuta šaka
10. **Open Hand** - Otvoren dlan

## Instalacija

### Zavisnosti

Instalirajte potrebne pakete iz `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Glavne biblioteke
- **TensorFlow** - za izgradnju i treniranje modela
- **OpenCV** - za rukovanje kamerom i slikama
- **MediaPipe** - za detekciju i landmarkove ruke
- **NumPy** - za numeričke operacije
- **scikit-learn** - za podjelu podataka

## Struktura projekta

```
├── data/
│   └── raw/                 # JSON datoteke s anotiranim uzorcima
├── models/                  # Spremljeni modeli i metapodaci
│   ├── hand_gesture_tf.keras
│   └── labels.json
├── train_model.py          # Skripta za prikupljanje podataka i treniranje modela
├── test_model.py           # Skripta za testiranje i inferenciju
├── requirements.txt        # Python zavisnosti
└── README.md              # Ovaj dokument
```

## Korištenje

### Pokretanje train_model.py

Glavna skripta omogućuje dva načina rada:

```bash
python train_model.py
```

Zatim odaberite opciju:
- **1** - Prikupljanje podataka
- **2** - Treniranje modela

### 1. Prikupljanje podataka

Kada odaberete opciju 1, skripta će pokrenuti kameru za prikupljanje uzoraka:

**Kontrole:**
- **0-9** - Odabir trenutne labele gesta (0 = OK, 1 = Finger gun, itd.)
- **s** - Spremi trenutni uzorak (mora biti detektirana ruka)
- **q** - Prekid prikupljanja

**Proces:**
1. Postavite ruku pred kameru
2. Pritisnite broj za odabir gesta koji izvodite
3. Pritisnite 's' za spremanje uzorka (preporučuje se 80 uzoraka po klasi)
4. Pauza između spremanja je 0.1 sekunde

Uzorci se automatski spremaju u `data/raw/` kao JSON datoteke s formatom:

```json
{
  "label": "OK",
  "landmarks": [x1, y1, z1, x2, y2, z2, ...],
  "handedness": "Left"
}
```

**Napomene o prikupljanju:**
- Landmarks niz sadrži 63 vrijednosti (21 točaka × 3 koordinate)
- Handedness se automatski zrcali zbog kamere (Right postaje Left i obrnuto)
- Preporučuje se prikupiti 80 uzoraka po klasi za dobru performansu

### 2. Treniranje modela

Kada odaberete opciju 2, skripta će:
- Učitati sve JSON datoteke iz `data/raw/`
- Normalizirati landmark koordinate (samo x,y koordinate se koriste)
- Podijeliti podatke na trening i validaciju (80/20)
- Trenirati MLP neuronsku mrežu
- Spremiti model u `models/hand_gesture_tf.keras` i labele u `models/labels.json`

### 3. Testiranje modela

Za testiranje koristite `test_model.py` (ako postoji).

## Model arhitektura

Model koristi MLP (Multi-Layer Perceptron) arhitekturu:

```
Input (42 značajke) -> BatchNorm -> Dense(128, ReLU) -> Dropout(0.3) -> 
Dense(64, ReLU) -> Dropout(0.3) -> Dense(10, Softmax)
```

### Značajke
- **Ulaz**: 42 normalizirane (x,y) koordinate za 21 landmark točku (z koordinata se zanemaruje)
- **Normalizacija**: Koordinate se skaliraju na [0,1] unutar bounding boxa ruke
- **Regularizacija**: Dropout slojevi (0.3) za sprečavanje preuučenja
- **Rani prekid**: Treniranje se zaustavlja nakon 10 epoha bez poboljšanja (patience=10)
- **Optimizer**: Adam s learning rate 1e-3
- **Loss funkcija**: SparseCategoricalCrossentropy

### Parametri treniranja
- **Epohe**: Do 100 (s ranim prekidom)
- **Batch size**: 32
- **Validation split**: 20%
- **Metric**: Accuracy

## Konfiguracija

Glavne postavke na vrhu `train_model.py`:

```python
GESTURE_LABELS = [...]        # Lista podržanih gesta
SAMPLES_PER_CLASS = 80       # Preporučeni broj uzoraka po klasi
SAVE_PAUSE_SEC = 0.1         # Pauza između spremanja uzoraka
TF_MODEL_PATH = "models/..."  # Putanja za spremanje modela
```

## Ograničenja

- Model radi s jednom rukom po kadru
- Zahtijeva dobro osvjetljenje za optimalnu detekciju
- Performanse ovise o kvaliteti i količini prikupljenih podataka
- Koriste se samo x,y koordinate (z se zanemaruje)
- Handedness se automatski zrcali zbog kamere

## Troubleshooting

**Problem**: "Kamera nije dostupna"
- Provjerite da li je kamera povezana i nije korištena od strane druge aplikacije
- Promijenite indeks kamere u `cv2.VideoCapture(0)` na 1 ili 2

**Problem**: "Nema prikupljenih podataka"
- Prvo pokrenite opciju 1 za prikupljanje podataka
- Provjerite postoje li JSON datoteke u `data/raw/` direktoriju

**Problem**: "Nema valjanih uzoraka nakon remapiranja"
- Provjerite da labele u JSON datotekama odgovaraju onima u `GESTURE_LABELS`
- Provjerite format JSON datoteka

**Problem**: Loša točnost modela
- Povećajte `SAMPLES_PER_CLASS` (preporučuje se 80+ uzoraka po klasi)
- Provjerite kvalitetu anotacija - gestikulirajte jasno i konzistentno
- Prikupite uzorke iz različitih uglova i uvjeta osvjetljenja
- Eksperimentirajte s arhitekturom modela

**Problem**: "Nema ruke" poruka tijekom prikupljanja
- Postavite ruku jasno pred kameru
- Povećajte `min_detection_confidence` parametar u MediaPipe Hands
- Provjerite osvjetljenje

## Napredne značajke

### Normalizacija koordinata
Funkcija `preprocess_landmarks_xy()` normalizira koordinate:
1. Pronalazi bounding box ruke (min/max x,y)
2. Skalira sve koordinate na [0,1] raspon unutar tog boxa
3. Vraća samo x,y koordinate (42 značajke umjesto 63)

### Stratified split
Kod koristi stratified podjelu podataka za balansiranu distribuciju klasa u train/val skupovima.

### Model callbacks
- **EarlyStopping**: Zaustavlja treniranje ako se val_accuracy ne poboljšava 10 epoha
- **restore_best_weights**: Vraća najbolje težine iz treniranja

## Licence

Ovaj projekt koristi otvorene biblioteke:
- TensorFlow (Apache 2.0)
- OpenCV (Apache 2.0)  
- MediaPipe (Apache 2.0)

---