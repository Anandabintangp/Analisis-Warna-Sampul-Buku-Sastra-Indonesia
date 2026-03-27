import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
from sklearn.cluster import KMeans
import cv2
import colorsys

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
COVER_DIR = "covers"
DATA_PATH = "src/data.csv"
MAX_COVERS = 48

WARNA_HEX = {
    "Merah": "#e63946",
    "Oranye/Cokelat": "#e07a5f",
    "Kuning": "#f4d35e",
    "Hijau": "#3d9970",
    "Biru": "#4895ef",
    "Ungu/Merah Muda": "#c77dff",
    "Hitam": "#1a1a1a",
    "Abu-abu/Putih": "#aaaaaa",
    "Lainnya": "#dddddd",
}
WARNA_ORDER = list(WARNA_HEX.keys())

# ─────────────────────────────────────────
# STREAMLIT SETUP
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Warna Sampul Sastra Indonesia",
    page_icon="📚",
    layout="wide"
)

# ─────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)

    cols = ["RATING_AVG","TAHUN_TERBIT","warna_r","warna_g","warna_b","TOTAL_RATINGS"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["warna_kategori"] = df.get("warna_kategori","Lainnya").fillna("Lainnya")
    return df


def get_cover_path(filename):
    return os.path.join(COVER_DIR, str(filename))


def extract_palette(image_path, k=5):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (80, 80))
        pixels = img.reshape(-1, 3)

        kmeans = KMeans(n_clusters=k, n_init=5)
        labels = kmeans.fit_predict(pixels)

        colors = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(labels)
        percents = counts / counts.sum()

        palette = sorted(
            [("#{:02x}{:02x}{:02x}".format(*c), p) for c, p in zip(colors, percents)],
            key=lambda x: x[1],
            reverse=True
        )
        return palette
    except:
        return None


def map_color(hex_color):
    try:
        hex_color = hex_color.lstrip("#")
        r,g,b = tuple(int(hex_color[i:i+2],16)/255 for i in (0,2,4))
        h,s,v = colorsys.rgb_to_hsv(r,g,b)
        h = h*360

        if v < 0.2: return "Hitam"
        elif s < 0.2 and v > 0.8: return "Abu-abu/Putih"
        elif h < 20 or h > 340: return "Merah"
        elif h < 45: return "Oranye/Cokelat"
        elif h < 70: return "Kuning"
        elif h < 160: return "Hijau"
        elif h < 260: return "Biru"
        else: return "Ungu/Merah Muda"
    except:
        return "Lainnya"


@st.cache_data
def load_palettes(df):
    palettes = {}
    for i,row in df.iterrows():
        path = get_cover_path(row["NAMA_FILE_GAMBAR"])
        if os.path.exists(path):
            palettes[i] = extract_palette(path)
        else:
            palettes[i] = None
    return palettes


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = load_data()

if df is None:
    st.error("Data tidak ditemukan")
    st.stop()

# popularity score
C = df["RATING_AVG"].mean()
m = df["TOTAL_RATINGS"].quantile(0.75)

def pop_score(row):
    if pd.isna(row["TOTAL_RATINGS"]) or pd.isna(row["RATING_AVG"]):
        return np.nan
    v = row["TOTAL_RATINGS"]
    R = row["RATING_AVG"]
    return (v/(v+m))*R + (m/(v+m))*C

df["POPULARITY_SCORE"] = df.apply(pop_score, axis=1)

# palettes
palette_dict = load_palettes(df)

# warna kategori dari gambar
new_cat = []
for i in df.index:
    p = palette_dict.get(i)
    if p:
        new_cat.append(map_color(p[0][0]))
    else:
        new_cat.append("Lainnya")

df["warna_kategori"] = new_cat

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    sort_option = st.selectbox(
        "Urutkan",
        ["Judul","Rating","Tahun","Populer"]
    )
    search = st.text_input("Cari judul")

# filter
fdf = df.copy()

if search:
    fdf = fdf[fdf["JUDUL"].str.contains(search, case=False, na=False)]

if sort_option=="Judul":
    fdf = fdf.sort_values("JUDUL")
elif sort_option=="Rating":
    fdf = fdf.sort_values("RATING_AVG", ascending=False)
elif sort_option=="Tahun":
    fdf = fdf.sort_values("TAHUN_TERBIT", ascending=False)
elif sort_option=="Populer":
    fdf = fdf.sort_values("POPULARITY_SCORE", ascending=False)

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("📚 Analisis Warna Sampul Buku")

st.write(f"Jumlah buku: {len(fdf)}")

subset = fdf.head(24)

cols = st.columns(6)

for i,(idx,row) in enumerate(subset.iterrows()):
    with cols[i%6]:
        path = get_cover_path(row["NAMA_FILE_GAMBAR"])

        if os.path.exists(path):
            st.image(path)
        else:
            st.write("No image")

        # palette
        palette = palette_dict.get(idx)

        if palette:
            bar_html = ""
            for hex_color, pct in palette:
                bar_html += f'<div style="width:{pct*100}%;height:10px;background:{hex_color}"></div>'
            st.markdown(f'<div style="display:flex">{bar_html}</div>', unsafe_allow_html=True)

        # info
        r = row["RATING_AVG"]
        p = row["POPULARITY_SCORE"]

        r_text = f"{r:.2f}" if pd.notna(r) else "-"
        p_text = f"{p:.2f}" if pd.notna(p) else "-"

        st.caption(f"{row['JUDUL'][:25]} \n⭐ {r_text} | 🔥 {p_text}")
