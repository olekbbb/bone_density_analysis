# Bone Density Analysis

## Uruchomienie frontendu

1. Otwórz terminal w folderze projektu:
   ```
   cd C:\Users\olasu\bone-density\bone_density_analysis
   ```

2. Uruchom serwer HTTP:
   ```
   python -m http.server 8000
   ```

3. Otwórz przeglądarkę i wejdź na:
   ```
   http://localhost:8000/frontend/
   ```

Żeby zatrzymać serwer — `Ctrl+C` w terminalu.

## Uruchomienie obliczeń

```
python -m obliczenia
```

Wyniki zapisują się do `wyniki_biomechaniczne.csv`.
