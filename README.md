# Hand Gesture Recognition

Projekt za prepoznavanje gesta ruke korištenjem 2D landmarkova iz MediaPipe biblioteke i TensorFlow neuronskih mreža.

## Pregled

Ovaj sustav omogućuje:
- Treniranje modela za prepoznavanje 10 različitih gesta ruke
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
├── train_model.py          # Skripta za treniranje modela
├── test_model.py           # Skripta za testiranje i inferenciju
├── requirements.txt        # Python zavisnosti
└── README.md              # Ovaj dokument
```

## Korištenje

### 1. Prikupljanje podataka

Prvo trebate prikupiti anotirane uzorke u `data/raw/` direktorij. Svaki uzorak treba biti JSON datoteka s formatom:

```json
{
  "label": "OK",
  "landmarks": [x1, y1, z1, x2, y2, z2, ...],
  "handedness": "Left"
}
```

Landmarks niz sadrži 63 vrijednosti (21 točaka × 3 koordinate).

### 2. Treniranje modela

Pokrenite treniranje:

```bash
python train_model.py
```

Skripta će:
- Učitati sve JSON datoteke iz `data/raw/`
- Normalizirati landmark koordinate
- Podijeliti podatke na trening i validaciju (80/20)
- Trenirati MLP neuronsku mrežu
- Spremiti model u `models/hand_gesture_tf.keras`

### 3. Testiranje modela

Za brzu provjeru modela:

```bash
python test_model.py
```

Za real-time inferenciju s kamerom, odkomentirajte liniju u `test_model.py`:

```python
# run_realtime_inference()  # Odkomentiraj ovu liniju
```

## Model arhitektura

Model koristi jednostavnu MLP (Multi-Layer Perceptron) arhitekturu:

```
Input (42 značajke) -> BatchNorm -> Dense(128, ReLU) -> Dropout(0.3) -> 
Dense(64, ReLU) -> Dropout(0.3) -> Dense(10, Softmax)
```

### Značajke
- **Ulaz**: 42 normalizirane (x,y) koordinate za 21 landmark točku
- **Normalizacija**: Koordinate se skaliraju na [0,1] unutar bounding boxa ruke
- **Regularizacija**: Dropout slojevi za sprečavanje preučenja
- **Rani prekid**: Treniranje se zaustavlja kada validacijska točnost prestane rasti

## Real-time inferencija

Real-time mode koristi:
- **OpenCV** za pristup kameri
- **MediaPipe Hands** za detekciju ruke i landmarkova
- **TensorFlow model** za klasifikaciju gesta

### Kontrole
- Pritisnite `q` za izlaz iz real-time moda
- Model prikazuje predikciju samo ako je pouzdanost >= 0.6

## Konfiguracija

Glavne postavke mogu se mijenjati na vrhu skripti:

```python
GESTURE_LABELS = [...]        # Lista podržanih gesta
NUM_CLASSES = len(TF_LABELS) # Broj klasa
TF_MODEL_PATH = "models/..."  # Putanja za spremanje modela
```

## Ograničenja

- Model radi s jednom rukom 
- Zahtijeva dobro osvjetljenje za optimalnu detekciju
- Performanse ovise o kvaliteti prikupljenih podataka
- Normalizacija koordinata može utjecati na rotacijske varijante istog gesta

## Troubleshooting

**Problem**: "Kamera nije dostupna"
- Provjerite da li je kamera povezana
- Promijenite indeks kamere u `cv2.VideoCapture(0)` na 1 ili 2

**Problem**: "Nema spremljenog modela"
- Prvo pokrenite `train_model.py` za treniranje
- Provjerite postoji li `models/hand_gesture_tf.keras`

**Problem**: Loša točnost modela
- Povećajte broj uzoraka po klasi
- Provjerite kvalitetu anotacija u JSON datotekama
- Eksperimentirajte s parametrima modela

## Licence

Ovaj projekt koristi otvorene biblioteke:
- TensorFlow (Apache 2.0)
- OpenCV (Apache 2.0)  
- MediaPipe (Apache 2.0)

---

