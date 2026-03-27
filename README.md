---
title: Analisis Warna Sampul Buku
emoji: 📚
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---

# 📚 Analisis Warna Sampul Fiksi Sastra Indonesia

Aplikasi visualisasi interaktif untuk menganalisis warna dominan pada sampul
1.229 novel fiksi sastra Indonesia yang dikumpulkan dari Goodreads.

## Fitur

- **Grid Sampul** — telusuri sampul dengan filter judul, penulis, genre, tahun, dan rating
- **Tren Warna per Tahun** — lihat bagaimana distribusi palet warna berubah dari 2000–2025
- **Analisis Agregat** — donut chart, scatter HSV, rating per warna, dan warna per genre

## Metodologi Warna

Warna diekstrak menggunakan KMeans (k=5) pada ruang warna RGB.
Kategorisasi mengikuti konvensi Jeong (2015/2017) dan ImagePlot (Manovich).
Parameter: Hue (H), Saturation (S), Brightness/Value (V) dalam skala HSV.

## Struktur File

```
├── src/
│   └── streamlit_app.py    # aplikasi utama
├── covers/                 # folder gambar sampul (unggah secara terpisah)
│   └── *.jpg
├── data.csv                # dataset hasil analisis
├── requirements.txt
└── README.md
```

## Cara Upload Sampul

Karena gambar terlalu banyak untuk Git, unggah folder `covers/` via:
```bash
git lfs track "covers/*.jpg"
git add .gitattributes covers/
```
Atau gunakan Hugging Face dataset sebagai sumber gambar.

## Referensi

- Jeong, W. (2015). *Media Visualization of Book Cover Images.* ASIST 2015.
- Manovich, L. (2020). *Cultural Analytics.* MIT Press.
- Genette, G. (1997). *Paratexts: Thresholds of Interpretation.* Cambridge UP.