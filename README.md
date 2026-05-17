# Bone Density Analysis

Projekt do segmentacji stosu obrazow TIFF kosci, obliczania parametrow biomechanicznych oraz lokalnej wizualizacji wynikow.

## Przygotowanie

Upewnij sie, ze obrazy TIFF sa w domyslnym katalogu:

```text
data/bone_34_476/*.tif
```

Mozesz tez podac inny wzorzec plikow przez `--input-glob`.

## Pierwsze uruchomienie segmentacji

Pelne uruchomienie liczy segmentacje, zapisuje gotowa segmentacje do pliku `.npz`, liczy wyniki biomechaniczne i na koncu otwiera viewer Napari:

```bash
.venv/bin/python segmentacja.py
```

Domyslnie powstaja pliki:

```text
outputs/gotowa_segmentacja.npz
outputs/wyniki_biomechaniczne.csv
outputs/wyniki_data.json
```

Jesli chcesz policzyc wszystko bez otwierania Napari:

```bash
.venv/bin/python segmentacja.py --skip-viewer
```

## Tryb oszczedzania RAM

Etap czyszczenia po `Segmentacja strukturalna` jest liczony blokami po osi Z. Zalewanie bryly domyslnie dziala w trybie `2d`, czyli przekroj po przekroju, zeby nie tworzyc duzych struktur 3D w RAM.

```bash
.venv/bin/python segmentacja.py --skip-viewer
```

Jesli proces dostaje `Killed`, zmniejsz rozmiar bloku czyszczenia:

```bash
.venv/bin/python segmentacja.py --skip-viewer --block-depth 24 --block-overlap 8
```

Najbardziej oszczedny wariant:

```bash
.venv/bin/python segmentacja.py --skip-viewer --block-depth 12 --block-overlap 4 --fill-mode 2d --fill-radius 30
```

Znaczenie opcji:

- `--block-depth` - ile przekrojow Z jest przetwarzanych naraz,
- `--block-overlap` - nakladka miedzy blokami dla czyszczenia glownej kosci,
- `--fill-mode 2d` - najbezpieczniejsze pamieciowo zalewanie bryly przekroj po przekroju,
- `--fill-radius` - promien domykania przy zalewaniu bryly.

Stary wariant blokowego zalewania 3D jest nadal dostepny:

```bash
.venv/bin/python segmentacja.py --skip-viewer --fill-mode 3d-block --block-depth 12 --block-overlap 4 --fill-overlap 8
```

Uzywaj go tylko do porownan, bo zuzywa znacznie wiecej RAM. Mniejszy blok zuzywa mniej RAM, ale moze liczyc sie dluzej i dawac slabsze laczenie na granicach blokow.

## Viewer bez ponownej segmentacji

Po pierwszym przeliczeniu mozna uruchomic sam viewer na zapisanej segmentacji:

```bash
.venv/bin/python segmentacja.py --viewer-only
```

To wczytuje `outputs/gotowa_segmentacja.npz` i pomija etap segmentacji oraz obliczen.

Jesli zapis segmentacji jest w innym pliku:

```bash
.venv/bin/python segmentacja.py --viewer-only --segmentation-file sciezka/do/segmentacji.npz
```

## Wymuszenie ponownej segmentacji

Jesli chcesz przeliczyc segmentacje od zera mimo istnienia `outputs/gotowa_segmentacja.npz`:

```bash
.venv/bin/python segmentacja.py --force-segmentation
```

Bez otwierania viewera:

```bash
.venv/bin/python segmentacja.py --force-segmentation --skip-viewer
```

## Dashboard wynikow

Po analizie skrypt zapisuje dane dla frontendu do:

```text
outputs/wyniki_data.json
```

Uruchom lokalny serwer statyczny:

```bash
.venv/bin/python -m http.server 8000 --directory .
```

Nastepnie otworz w przegladarce:

```text
http://127.0.0.1:8000/frontend/wyniki.html
```

Dashboard pokazuje:

- podstawowe metryki srednia/min/max,
- wykres metryki po przekrojach,
- wykres porownawczy dwoch metryk,
- tabele wynikow.

Mozesz tez otworzyc `frontend/wyniki.html` bez serwera i recznie wczytac `outputs/wyniki_biomechaniczne.csv` przyciskiem `Wczytaj CSV`.

## Przydatne opcje

Zmiana lokalizacji danych wejsciowych:

```bash
.venv/bin/python segmentacja.py --input-glob "data/inny_folder/*.tif"
```

Zmiana pliku zapisanej segmentacji:

```bash
.venv/bin/python segmentacja.py --segmentation-file outputs/probka_01.npz
```

Zmiana ROI:

```bash
.venv/bin/python segmentacja.py --roi 100 600 100 600
```

Zmiana rozmiaru blokow dla slabiej dostepnej pamieci:

```bash
.venv/bin/python segmentacja.py --block-depth 24 --block-overlap 8 --fill-mode 2d --fill-radius 30
```

Zmiana lokalizacji danych dashboardu:

```bash
.venv/bin/python segmentacja.py --dashboard-data outputs/wyniki_data.json
```

## Typowy workflow

1. Policz segmentacje i wyniki:

```bash
.venv/bin/python segmentacja.py --skip-viewer
```

2. Obejrz segmentacje w Napari pozniej:

```bash
.venv/bin/python segmentacja.py --viewer-only
```

3. Obejrz wyniki w dashboardzie:

```bash
.venv/bin/python -m http.server 8000 --directory .
```

```text
http://127.0.0.1:8000/frontend/wyniki.html
```
