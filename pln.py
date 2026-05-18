import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.title("VERSI BARU 100%")

st.write("Lokasi file yang sedang berjalan:")
st.write(os.getcwd())

st.set_page_config(layout="wide")

st.title("Dashboard Permohonan Listrik")
st.subheader("Pasang Baru & Tambah Daya")

# ========================
# LOAD DATA
# ========================
df = pd.read_csv("PBPD 2025 MAGANG.csv")

st.success("Data berhasil dibaca")
st.write("Jumlah Data:", len(df))

# ========================
# KONVERSI TIPE DATA
# ========================
df["TGLMOHON"] = pd.to_datetime(df["TGLMOHON"], errors="coerce")
df["BULAN"] = df["TGLMOHON"].dt.to_period("M")

df["DAYA_LAMA"] = pd.to_numeric(df["DAYA_LAMA"], errors="coerce")
df["DAYA"] = pd.to_numeric(df["DAYA"], errors="coerce")

df["SELISIH_DAYA"] = df["DAYA"] - df["DAYA_LAMA"]

# ========================
# KPI
# ========================
st.subheader("KPI Utama")

col1, col2, col3 = st.columns(3)

col1.metric("Total Permohonan", len(df))

if "JENIS_TRANSAKSI" in df.columns:
    col2.metric(
        "Jenis Terbanyak",
        df["JENIS_TRANSAKSI"].value_counts().idxmax()
    )

col3.metric(
    "Rata-rata Selisih Daya",
    round(df["SELISIH_DAYA"].mean(), 2)
)

# ========================
# TREN BULANAN
# ========================
st.subheader("Tren Permohonan Bulanan")

tren = df.groupby("BULAN").size()

fig, ax = plt.subplots()
ax.plot(tren.index.astype(str), tren.values)
plt.xticks(rotation=45)
st.pyplot(fig)

# ========================
# DISTRIBUSI JENIS
# ========================
st.subheader("Distribusi Jenis Transaksi")

jenis = df["JENIS_TRANSAKSI"].value_counts()

fig2, ax2 = plt.subplots()
ax2.bar(jenis.index, jenis.values)
plt.xticks(rotation=45)
st.pyplot(fig2)

# ========================
# TABEL DATA
# ========================
st.subheader("Preview Data")
st.dataframe(df.head())