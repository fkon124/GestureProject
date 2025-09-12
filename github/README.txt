# README.txt

## Opis projekta
Sustav za prepoznavanje gesta rukom temeljen na MediaPipe (detekcija ruke/landmarka), OpenCV (kamera) i TensorFlow (klasifikacija). Projekt omogućuje:
- prikupljanje podataka (landmark točke ruke) i spremanje u `data/raw/` kao JSON,
- treniranje TensorFlow modela (`models/hand_gesture_tf.keras` + `models/labels.json`),
- real-time prepoznavanje gesta preko kamere.

---

## Autori
Ovaj projekt su, uz mentora Juraja Benića, izradili studenti Fakulteta primijenjene matematike i informatike u Osijeku, Luka Tomljanović i Filip Kon.

---

## Preduvjeti i instalacija

Preporučeno okruženje (Windows): Python 3.11 i virtualno okruženje.

1) Napravite i aktivirajte venv (PowerShell):
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2) Instalirajte ovisnosti:
```powershell
python -m pip install -r requirements.txt
```

Ako Mediapipe nedostaje ili javlja grešku na krivoj verziji Pythona, instalirajte eksplicitno (za Py 3.11):
```powershell
python -m pip install mediapipe==0.10.21 opencv-python
```

Ako dođe do konflikta TensorFlowa i JAX-a (ml_dtypes greške):
```powershell
python -m pip uninstall -y jax jaxlib ml-dtypes
python -m pip install "ml_dtypes>=0.3.1,<0.4" tensorflow==2.16.2
```
Nakon toga restartajte Jupyter kernel/IDE interpreter.

Napomena (Apple Silicon):
```bash
pip install -U tensorflow-macos tensorflow-metal
```

---

## Struktura projekta

- `projektGesta.ipynb` – glavni notebook: postavke, prikupljanje, treniranje, inferencija.
- `data/raw/` – prikupljeni uzorci u JSON formatu: `{label, handedness, landmarks}`.
- `models/` – spremljeni modeli i metapodaci:
  - `hand_gesture_tf.keras` – TensorFlow model,
  - `labels.json` – lista labela korištena u modelu.

---

## Rad u Jupyteru (preporučeni redoslijed)

1) Setup: ćelija s varijablama `GESTURE_LABELS`, `TF_LABELS`, putanjama (`RAW_DIR`, `TF_MODEL_PATH`, ...), te importima (`tensorflow as tf`, `train_test_split`, ...). Pokrenuti prva.
2) Funkcija normalizacije: ćelija s `preprocess_landmarks_xy` (centiranje na zglob, opcionalno zrcaljenje desne ruke, skaliranje).
3) Prikupljanje: pokrenuti ćeliju „Prikupljanje podataka”, tipke `0–9/...` biraju labelu, `s` sprema uzorak, `q` izlaz.
4) Treniranje: pokrenuti ćeliju „Treniranje TF modela (s normalizacijom)”. Model i `labels.json` se spremaju u `models/`.
5) Učitavanje + real-time: pokrenuti ćelije za učitavanje modela i real-time inferenciju.

Minimalni skup ćelija koje trebaju proći nakon restarta kernela: (1) Setup, (2) Normalizacija, (4) Treniranje ili (5) Učitavanje, zatim (5) Inferencija.

---

## Normalizacija značajki (važno)
Koristi se funkcija `preprocess_landmarks_xy` koja:
- izdvaja XY koordinate,
- po želji zrcali desnu ruku (smanjuje varijacije orijentacije),
- translatira na zglob (wrist) kao ishodište,
- skalira na jediničnu veličinu po maksimalnoj udaljenosti od ishodišta.

Ova normalizacija u praksi znatno poboljšava točnost i robusnost u odnosu na sirove XY koordinate.

---

## Prikupljanje podataka

1. Pokrenite sekciju „Prikupljanje podataka”.
2. `0–9` biraju labelu, `s` sprema trenutni uzorak, `q` izlaz.
3. Svaki uzorak: 21 landmark × (x,y,z) + `handedness`. Pohrana: `data/raw/*.json`.
4. Parametar `SAMPLES_PER_CLASS` kontrolira željeni broj po klasi.

Savjet: Ciljajte barem 20–50 uzoraka po klasi za stabilan model.

---

## Treniranje modela (notebook)

- Ćelija „Treniranje TensorFlow modela s normalizacijom” učitava JSON uzorke iz `data/raw/`, pretvara u normalizirani XY (42D), radi podjelu na train/val te trenira Keras model. 
- Model: `models/hand_gesture_tf.keras`, labele: `models/labels.json`.

Mali skupovi podataka: Ako imate npr. 3 uzorka/klasa, koristite veći `test_size` (tako da test ima ≥1 po klasi) ili ručno izdvojite 1 uzorak po klasi u validaciju.

---

## Real-time prepoznavanje

- U notebooku: pokrenuti ćeliju „Učitavanje modela”, zatim „Real-time inferencija (TF)”.
- U skripti: `python test_model.py` (provjerite da je aktivno okruženje s ispravnim Python interpreterom).

Tipke: prozor se zatvara na `q`.

---

## Pokretanje iz terminala (Windows)

Otvorite terminal u mapi projekta i aktivirajte venv:
```powershell
cd D:\FaksProgramiranje\ProjektGesta
.\.venv311\Scripts\Activate.ps1
python train_model.py
python test_model.py
```

VS Code / Cursor: Ctrl+` → Terminal; Ctrl+Shift+P → Python: Select Interpreter → odaberite `.venv311`.

---

## Česta pitanja i rješavanje problema (FAQ)

- „No module named mediapipe” ili nema wheel-a:
  - Koristite Python 3.11. Instalirajte: `python -m pip install mediapipe==0.10.21 opencv-python`.

- TensorFlow se ne može importati, greška oko `ml_dtypes`/JAX:
  - Uklonite JAX i zaključajte ml_dtypes:  
    `pip uninstall -y jax jaxlib ml-dtypes`  
    `pip install "ml_dtypes>=0.3.1,<0.4" tensorflow==2.16.2`

- `NameError: RAW_DIR/GESTURE_LABELS/TF_LABEL_REMAP/preprocess_landmarks_xy` nije definiran:
  - Pokrenite setup ćeliju (postavke i labele) i ćeliju s normalizacijom prije treninga/inferencije.

- Greške sa stratificiranim splitom: „least populated class = 1”, „test_size < num_classes”:
  - Povećajte `test_size` (tako da test ima ≥1 uzorak po klasi),
  - ili uklonite stratifikaciju, 
  - ili privremeno filtrirajte vrlo rijetke klase / prikupite više uzoraka.

- „Permission denied” pri kreiranju venv-a:
  - Zatvorite procese Pythona, obrišite neispravan venv i napravite venv u čistoj lokaciji (npr. `D:\venvs\gesta311`).

---

## Napomene

- Potrebna je funkcionalna kamera.
- Preporučeni Python: 3.11. 
- Uvijek prvo pokrenite setup ćelije u notebooku nakon restarta kernela.
- Ako mijenjate skup gesta, uredite `GESTURE_LABELS` i ponovno istrenirajte model.
