# README.txt

## Opis projekta
Ovaj projekt implementira sustav za prepoznavanje gesta rukom koristeći MediaPipe za detekciju ruku i OpenCV za rad s kamerom.  
Projekt omogućuje:
- prikupljanje podataka (landmark točke ruke) i njihovo spremanje,
- treniranje modela strojnog učenja (npr. klasifikator iz scikit-learn),
- pokretanje real-time prepoznavanja gesta preko kamere.

---

## Potrebne biblioteke

Za pokretanje projekta potrebno je instalirati sljedeće pakete:

```bash
pip install mediapipe opencv-python scikit-learn joblib matplotlib seaborn
```

Ako koristite Apple Silicon (M1/M2/M3), možete instalirati optimizirani TensorFlow:

```bash
pip install -U tensorflow-macos tensorflow-metal
```

---

## Struktura projekta

- `projektGesta.ipynb` – glavni Jupyter notebook koji sadrži kod za:
  - pripremu i instalaciju paketa,
  - prikupljanje podataka gesta,
  - treniranje modela,
  - testiranje i vizualizaciju rezultata,
  - real-time prepoznavanje.

- `data/` – mapa u kojoj se spremaju prikupljeni podaci o gestama (CSV datoteke).  
- `models/` – mapa za spremljene modele treniranja (npr. `.joblib`).  

---

## Prikupljanje podataka

1. Pokrenite notebook i odaberite sekciju Prikupljanje podataka.  
2. Pritiskom na tipke 0–11 odabirete labelu geste koju snimate.  
3. Pritiskom na s spremate trenutni uzorak.  
4. Pritiskom na q izlazite iz aplikacije.  
5. Svaka gesta se sprema kao 63 značajke (21 landmark × 3 koordinate) + handedness.  

Parametar `SAMPLES_PER_CLASS` određuje koliko uzoraka po gesti želite prikupiti.  

---

## Treniranje modela

1. Nakon prikupljanja podataka, pokrenite sekciju Treniranje modela.  
2. Koristi se scikit-learn klasifikator (npr. `RandomForestClassifier`).  
3. Model se sprema pomoću `joblib` u mapu `models/`.  


## Prepoznavanje gesta u realnom vremenu

1. Pokrenite sekciju Real-time prepoznavanje.  
2. Kamera se uključuje pomoću OpenCV (`cv2.VideoCapture`).  
3. MediaPipe detektira landmark točke ruke.  
4. Model predviđa kojoj gesti pripada trenutni uzorak.  
5. Predikcija se ispisuje i prikazuje na ekranu.  

---

## Vizualizacija

Projekt koristi `matplotlib` i `seaborn` za:
- prikaz distribucije podataka po klasama,  
- iscrtavanje matrice konfuzije,  
- evaluaciju performansi modela.  

---

## Napomene

- Potrebna je funkcionalna kamera (laptop/web kamera).  
- Okruženje: Python 3.9+ preporučeno.  
- Ako koristite Jupyter Notebook, pokrenite sve ćelije redom.  
- Ako želite prilagoditi broj klasa gesta, uredite varijablu `GESTURE_LABELS`.  
