# Diabetter ML

## Get the Model

You need a Linux operating system to run the model. The model is approximately 700 MB in size, so we have placed it at [Google Drive](https://drive.google.com/file/d/1YwtSaSc5C4jDuOANtfTqKDaqrOzdMMDd/view?usp=sharing).

## Run The Model

Run the model can be done using terminal. There are several parameters are required.

```bash
./modelv2 -h
usage: modelv2 [-h] [--berat-badan BERAT_BADAN] [--tinggi TINGGI] [--usia USIA] [--jenis-kelamin JENIS_KELAMIN] [--aktivitas AKTIVITAS] [--filter FILTER]

options:
  -h, --help            show this help message and exit
  --berat-badan BERAT_BADAN
  --tinggi TINGGI
  --usia USIA
  --jenis-kelamin JENIS_KELAMIN
  --aktivitas AKTIVITAS
  --filter FILTER
```

1. `--berat-badan` integer
2. `--tinggi` integer
3. `--usia` integer
4. `--jenis-kelamin` pria | wanita
5. `--aktivitas` ringan | sedang | berat
6. `--filter` float as x, 0.0 <= x <= 5.0

Example:

```bash
./modelv2 --berat-badan=60 --tinggi=160 --usia=22 --jenis-kelamin=pria --aktivitas=sedang --filter=4.0
```

Result:

```bash
Total Kalori: 2025.0 kalori per hari
Kebutuhan Karbohidrat: 253.12 gram per hari
Kebutuhan Protein: 101.25 gram per hari
Kebutuhan Lemak: 56.25 gram per hari

Rekomendasi makanan dan kandungan gizinya:
Pangasius Goreng - Kalori: 572.0, Karbohidrat: 4.6, Protein: 97.0, Lemak: 18.4, Rating: 4.55
Mie Goreng - Kalori: 936.0, Karbohidrat: 124.8, Protein: 15.2, Lemak: 40.8, Rating: 4.19
Getuk Lindri - Kalori: 342.8, Karbohidrat: 64.0, Protein: 8.0, Lemak: 8.0, Rating: 4.75

Total Kalori dari makanan yang direkomendasikan: 2108.6
Total Karbohidrat dari makanan yang direkomendasikan: 247.6
Total Protein dari makanan yang direkomendasikan: 50.8
Total Lemak dari makanan yang direkomendasikan: 82.4
```
