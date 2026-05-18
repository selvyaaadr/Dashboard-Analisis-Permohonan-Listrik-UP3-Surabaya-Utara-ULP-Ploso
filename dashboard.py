import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Supaya tampilan grafik lebih rapi
plt.style.use('default')

# Load data 
df = pd.read_csv("PBPD 2025 MAGANG.csv")

# Lihat 5 data pertama
df.head()

# DATA CLEANING 

# cek data
df.info()

# ubah format tanggal 
df['TGLMOHON'] = pd.to_datetime(df['TGLMOHON'])

# tambah kolom tahun dan bulan 
df['tahun'] = df['TGLMOHON'].dt.year
df['bulan'] = df['TGLMOHON'].dt.month
df['nama_bulan'] = df['TGLMOHON'].dt.strftime('%B')

# kolom selisi daya 
df['DAYA_LAMA'] = df['DAYA_LAMA'].fillna(0)
df['selisih_daya'] = df['DAYA'] - df['DAYA_LAMA']

# RAPI NAMA KOLOM
df.columns = df.columns.str.strip()   # hapus spasi tersembunyi
df.columns = df.columns.str.lower()   # ubah jadi huruf kecil

print(df.columns)

# rename nama kolom
df = df.rename(columns={
    'tglmohon': 'tanggal_permohonan',
    'idpel': 'id_pelanggan',
    'nama': 'nama_pelanggan',
    'alamat': 'alamat_pelanggan',
    'tarif': 'tarif_baru',
    'daya': 'daya_baru'
})

df.columns

# UBAH FORMAT TANGGAL
df['tanggal_permohonan'] = pd.to_datetime(df['tanggal_permohonan'], errors='coerce')

df.info()

# KOLOM NUMERIK BERSIH
df['daya_lama'] = pd.to_numeric(df['daya_lama'], errors='coerce')
df['daya_baru'] = pd.to_numeric(df['daya_baru'], errors='coerce')

# KOLOM SELISIS DAYA 
df['daya_lama'] = df['daya_lama'].fillna(0)

df['selisih_daya'] = df['daya_baru'] - df['daya_lama']

# KOLOM TAHUN & BULAN 
df['tahun'] = df['tanggal_permohonan'].dt.year
df['bulan'] = df['tanggal_permohonan'].dt.month
df['nama_bulan'] = df['tanggal_permohonan'].dt.strftime('%B')

# MISSING VALUE CEK 
df.isnull().sum()

# ANALISIS TEMPORAL 

# jumlah permohonan per bulan 
# Agregasi jumlah permohonan per bulan
permohonan_bulanan = (
    df
    .groupby(pd.Grouper(key='tanggal_permohonan', freq='M'))
    .size()
    .reset_index(name='jumlah_permohonan')
)

plt.figure()
plt.plot(permohonan_bulanan['tanggal_permohonan'],
         permohonan_bulanan['jumlah_permohonan'])

plt.title("Tren Permohonan per Bulan")
plt.xlabel("Waktu")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=45)
plt.show()

# jumlah permohonan per tahub
permohonan_tahunan = df.groupby('tahun').size().reset_index(name='jumlah')

plt.figure()
plt.plot(permohonan_tahunan['tahun'],
         permohonan_tahunan['jumlah'])

plt.title("Jumlah Permohonan per Tahun")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Permohonan")
plt.show()

# bulan dengan permohonan tertinggi
top_bulan = (
    df.groupby('nama_bulan')
      .size()
      .sort_values(ascending=False)
)

print(top_bulan)

plt.figure()
top_bulan.plot(kind='bar')
plt.title("Distribusi Permohonan per Bulan")
plt.xlabel("Bulan")
plt.ylabel("Jumlah")
plt.xticks(rotation=45)
plt.show()


# Moving Average
permohonan_bulanan['moving_avg'] = (
    permohonan_bulanan['jumlah_permohonan']
    .rolling(window=3)
    .mean()
)

plt.figure()
plt.plot(permohonan_bulanan['tanggal_permohonan'],
         permohonan_bulanan['jumlah_permohonan'])

plt.plot(permohonan_bulanan['tanggal_permohonan'],
         permohonan_bulanan['moving_avg'])

plt.title("Tren Permohonan + Moving Average")
plt.xlabel("Waktu")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=45)
plt.show()

# analisis jenis transaksi

# jumlah dan presentase 

import pandas as pd
import matplotlib.pyplot as plt

# Hitung jumlah per jenis transaksi
transaksi_counts = df['jenis_transaksi'].value_counts().reset_index()
transaksi_counts.columns = ['jenis_transaksi', 'jumlah']

# Hitung persentase
transaksi_counts['persentase (%)'] = (
    transaksi_counts['jumlah'] / transaksi_counts['jumlah'].sum() * 100
)

print(transaksi_counts)

plt.figure()
plt.bar(transaksi_counts['jenis_transaksi'],
        transaksi_counts['jumlah'])

plt.title("Distribusi Jenis Transaksi")
plt.xlabel("Jenis Transaksi")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=45)
plt.show()

plt.figure()
plt.pie(transaksi_counts['jumlah'],
        labels=transaksi_counts['jenis_transaksi'],
        autopct='%1.1f%%')

plt.title("Persentase Jenis Transaksi")
plt.show()

# analisis tren transaksi

transaksi_tahunan = (
    df.groupby(['tahun', 'jenis_transaksi'])
      .size()
      .unstack()
      .fillna(0)
)

print(transaksi_tahunan)

plt.figure(figsize=(6,4))

for kolom in transaksi_tahunan.columns:
    plt.plot(transaksi_tahunan.index,
             transaksi_tahunan[kolom],
             marker='o')

plt.title("Tren Jenis Transaksi per Tahun")
plt.xlabel("Tahun")
plt.ylabel("Jumlah")
plt.legend()
plt.show()

plt.figure(figsize=(6,4))

transaksi_tahunan.plot(kind='bar')

plt.title("Distribusi Jenis Transaksi Tahun 2025")
plt.xlabel("Tahun")
plt.ylabel("Jumlah")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# transaksi per bulan 

# jumlah permohonan per bulan 

import matplotlib.pyplot as plt

# Agregasi jumlah permohonan per bulan
permohonan_bulanan = (
    df.groupby('bulan')
      .size()
      .reset_index(name='jumlah_permohonan')
      .sort_values('bulan')
)

print(permohonan_bulanan)

# Plot
plt.figure(figsize=(8,5))
plt.plot(permohonan_bulanan['bulan'],
         permohonan_bulanan['jumlah_permohonan'],
         marker='o')

plt.title("Tren Permohonan per Bulan Tahun 2025")
plt.xlabel("Bulan")
plt.ylabel("Jumlah Permohonan")
plt.xticks(range(1,13))
plt.grid(True)
plt.show()

# Transaksi 

transaksi_bulanan = (
    df.groupby(['bulan', 'jenis_transaksi'])
      .size()
      .unstack()
      .fillna(0)
)

print(transaksi_bulanan)

plt.figure(figsize=(8,5))

for kolom in transaksi_bulanan.columns:
    plt.plot(transaksi_bulanan.index,
             transaksi_bulanan[kolom],
             marker='o')

plt.title("Tren Jenis Transaksi per Bulan Tahun 2025")
plt.xlabel("Bulan")
plt.ylabel("Jumlah")
plt.xticks(range(1,13))
plt.legend()
plt.grid(True)
plt.show()


# Permohonan tertinggi VS terendah 

bulan_tertinggi = permohonan_bulanan.loc[
    permohonan_bulanan['jumlah_permohonan'].idxmax()
]

bulan_terendah = permohonan_bulanan.loc[
    permohonan_bulanan['jumlah_permohonan'].idxmin()
]

print("Bulan Tertinggi:")
print(bulan_tertinggi)

print("\nBulan Terendah:")
print(bulan_terendah)


import matplotlib.pyplot as plt

# Agregasi jumlah permohonan per bulan
permohonan_bulanan = (
    df.groupby('bulan')
      .size()
      .reset_index(name='jumlah_permohonan')
      .sort_values('bulan')
)

# Cari bulan tertinggi & terendah
bulan_tertinggi = permohonan_bulanan.loc[
    permohonan_bulanan['jumlah_permohonan'].idxmax()
]

bulan_terendah = permohonan_bulanan.loc[
    permohonan_bulanan['jumlah_permohonan'].idxmin()
]

# Plot grafik
plt.figure(figsize=(8,5))
plt.plot(permohonan_bulanan['bulan'],
         permohonan_bulanan['jumlah_permohonan'],
         marker='o')

# Tandai bulan tertinggi
plt.scatter(bulan_tertinggi['bulan'],
            bulan_tertinggi['jumlah_permohonan'])

plt.text(bulan_tertinggi['bulan'],
         bulan_tertinggi['jumlah_permohonan'],
         f"  Tertinggi\n  {int(bulan_tertinggi['jumlah_permohonan'])}")

# Tandai bulan terendah
plt.scatter(bulan_terendah['bulan'],
            bulan_terendah['jumlah_permohonan'])

plt.text(bulan_terendah['bulan'],
         bulan_terendah['jumlah_permohonan'],
         f"  Terendah\n  {int(bulan_terendah['jumlah_permohonan'])}")

plt.title("Tren Permohonan per Bulan Tahun 2025")
plt.xlabel("Bulan")
plt.ylabel("Jumlah Permohonan")
plt.xticks(range(1,13))
plt.grid(True)
plt.tight_layout()
plt.show()

# analisis perubahan daya 

# Statistik selisi perubahan daya 

# Statistik dasar
print("Statistik Selisih Daya:")
print(df['selisih_daya'].describe())

# Berapa yang menaikkan daya (>0)
menaikkan = (df['selisih_daya'] > 0).sum()
total = len(df)

print("\nJumlah pelanggan menaikkan daya:", menaikkan)
print("Persentase menaikkan daya:", round((menaikkan/total)*100,2), "%")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.hist(df['selisih_daya'], bins=20)

plt.title("Distribusi Selisih Daya")
plt.xlabel("Selisih Daya (VA)")
plt.ylabel("Jumlah Pelanggan")
plt.grid(True)
plt.show()

# rata-rata kenaikan daya 

rata_bulanan = (
    df.groupby('bulan')['selisih_daya']
      .mean()
      .reset_index()
      .sort_values('bulan')
)

print(rata_bulanan)

plt.figure(figsize=(8,5))
plt.plot(rata_bulanan['bulan'],
         rata_bulanan['selisih_daya'],
         marker='o')

plt.title("Rata-rata Kenaikan Daya per Bulan Tahun 2025")
plt.xlabel("Bulan")
plt.ylabel("Rata-rata Selisih Daya (VA)")
plt.xticks(range(1,13))
plt.grid(True)
plt.show()

# heatmap

import pandas as pd

# Buat pivot table
heatmap_data = (
    df.groupby(['bulan', 'jenis_transaksi'])
      .size()
      .unstack()
      .fillna(0)
)

print(heatmap_data)

import matplotlib.pyplot as plt
import numpy as np

# Pastikan urut bulan
heatmap_data = heatmap_data.sort_index()

fig, ax = plt.subplots(figsize=(8,6))

# Tampilkan heatmap
im = ax.imshow(heatmap_data.values, aspect='auto')

# Atur label sumbu X
ax.set_xticks(np.arange(len(heatmap_data.columns)))
ax.set_xticklabels(heatmap_data.columns)

# Atur label sumbu Y
ax.set_yticks(np.arange(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)

# Tambahkan angka di dalam kotak
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        ax.text(j, i,
                int(heatmap_data.values[i, j]),
                ha='center',
                va='center',
                fontsize=9)

# Tambahkan colorbar
cbar = fig.colorbar(im)
cbar.set_label('Jumlah Permohonan')

ax.set_title("Heatmap Bulanan vs Jenis Transaksi Tahun 2025")
ax.set_xlabel("Jenis Transaksi")
ax.set_ylabel("Bulan")

plt.tight_layout()
plt.show()

# Growth Rate Bulanan
 
import pandas as pd
import matplotlib.pyplot as plt

# Total permohonan per bulan
permohonan_bulanan = (
    df.groupby('bulan')
      .size()
      .reset_index(name='jumlah')
      .sort_values('bulan')
)

print(permohonan_bulanan)


# Hitung growth rate
permohonan_bulanan['growth_%'] = (
    permohonan_bulanan['jumlah']
    .pct_change() * 100
)

print(permohonan_bulanan)

plt.figure(figsize=(8,5))

plt.plot(permohonan_bulanan['bulan'],
         permohonan_bulanan['growth_%'],
         marker='o')

plt.axhline(0)  # garis nol untuk pembanding

plt.title("Growth Rate Bulanan Permohonan Tahun 2025")
plt.xlabel("Bulan")
plt.ylabel("Pertumbuhan (%)")
plt.xticks(range(1,13))
plt.grid(True)
plt.tight_layout()
plt.show()

# deteksi lonjakan dan penurunan 

bulan_tertinggi = permohonan_bulanan.loc[
    permohonan_bulanan['growth_%'].idxmax()
]

bulan_terendah = permohonan_bulanan.loc[
    permohonan_bulanan['growth_%'].idxmin()
]

print("Lonjakan Tertinggi:")
print(bulan_tertinggi)

print("\nPenurunan Terbesar:")
print(bulan_terendah)


# analisis spasial 

# Ekstrak kecamatan dari alamat
df['kecamatan'] = (
    df['alamat_pelanggan']
    .str.split(',')
    .str[-3]      # ambil bagian sebelum kota
    .str.strip()
)

print(df[['alamat_pelanggan','kecamatan']].head())

# Jumlah per mohonan per kecamatan 
permohonan_kecamatan = (
    df.groupby('kecamatan')
      .size()
      .reset_index(name='jumlah_permohonan')
      .sort_values('jumlah_permohonan', ascending=False)
)

print(permohonan_kecamatan)

import folium

# Titik tengah Surabaya
peta = folium.Map(location=[-7.2575, 112.7521], zoom_start=12)

# Tambahkan marker untuk setiap kecamatan
for _, row in permohonan_kecamatan.iterrows():
    folium.Marker(
        location=[-7.25, 112.75],  # sementara titik pusat (nanti kita bisa buat lebih akurat)
        popup=f"{row['kecamatan']} : {row['jumlah_permohonan']}",
    ).add_to(peta)

peta
peta.save("peta_permohonan.html")

df['kecamatan'] = (
    df['kecamatan']
    .str.upper()
    .str.strip()
)

permohonan_kecamatan = (
    df.groupby('kecamatan')
      .size()
      .reset_index(name='jumlah')
)

print(permohonan_kecamatan.head())

kecamatan_fokus = [
    "GENTENG",
    "GUBENG",
    "KENJERAN",
    "MULYOREJO",
    "SIMOKERTO",
    "TAMBAKSARI"
]

df['kecamatan'] = df['kecamatan'].str.upper().str.strip()
df_fokus = df[df['kecamatan'].isin(kecamatan_fokus)]

print(df_fokus['kecamatan'].value_counts())


import matplotlib.pyplot as plt

jumlah_kecamatan = df_fokus['kecamatan'].value_counts().sort_values()

plt.figure()
jumlah_kecamatan.plot(kind='barh')
plt.title("Jumlah Permohonan per Kecamatan")
plt.xlabel("Jumlah Permohonan")
plt.ylabel("Kecamatan")
plt.tight_layout()
plt.show()

import folium

# Koordinat centroid perkiraan
koordinat = {
    "GENTENG": [-7.265, 112.747],
    "GUBENG": [-7.276, 112.758],
    "KENJERAN": [-7.235, 112.800],
    "MULYOREJO": [-7.275, 112.785],
    "SIMOKERTO": [-7.245, 112.750],
    "TAMBAKSARI": [-7.255, 112.760],
}

jumlah = df_fokus['kecamatan'].value_counts().reset_index()
jumlah.columns = ['kecamatan','total']

peta = folium.Map(location=[-7.26, 112.77], zoom_start=12)

for _, row in jumlah.iterrows():
    kec = row['kecamatan']
    total = row['total']
    
    if kec in koordinat:
        folium.CircleMarker(
            location=koordinat[kec],
            radius=total/200,
            popup=f"{kec}: {total} permohonan",
            fill=True
        ).add_to(peta)

peta.save("peta_6_kecamatan.html")
peta

# permohonan pemasangan baru vs perubahan daya 

df_fokus['jenis_transaksi'] = df_fokus['jenis_transaksi'].str.upper().str.strip()

df_fokus['jenis_transaksi'].value_counts()

import matplotlib.pyplot as plt

pivot_layanan = pd.crosstab(
    df_fokus['kecamatan'],
    df_fokus['jenis_transaksi']
)

pivot_layanan.plot(kind='bar')
plt.title("Permohonan Pasang Baru vs Tambah Daya per Kecamatan")
plt.xlabel("Kecamatan")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# temporal per bulan 

df_fokus['tanggal_permohonan'] = pd.to_datetime(df_fokus['tanggal_permohonan'])
df_fokus['bulan'] = df_fokus['tanggal_permohonan'].dt.to_period('M')

tren = (
    df_fokus.groupby(['bulan','jenis_transaksi'])
    .size()
    .unstack()
)

tren.plot()
plt.title("Tren Bulanan Pasang Baru vs Tambah Daya")
plt.xlabel("Bulan")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# peta 
import folium
import pandas as pd

# Bersihkan format
df_fokus['jenis_transaksi'] = df_fokus['jenis_transaksi'].str.upper().str.strip()

# Pivot jumlah
pivot = pd.crosstab(
    df_fokus['kecamatan'],
    df_fokus['jenis_transaksi']
).reset_index()

# Pastikan kolom ada
if 'PASANG BARU' not in pivot.columns:
    pivot['PASANG BARU'] = 0
if 'TAMBAH DAYA' not in pivot.columns:
    pivot['TAMBAH DAYA'] = 0

pivot['TOTAL'] = pivot['PASANG BARU'] + pivot['TAMBAH DAYA']

# Hitung persentase
pivot['% PASANG BARU'] = (pivot['PASANG BARU'] / pivot['TOTAL'] * 100).round(1)
pivot['% TAMBAH DAYA'] = (pivot['TAMBAH DAYA'] / pivot['TOTAL'] * 100).round(1)

# Koordinat centroid
koordinat = {
    "GENTENG": [-7.265, 112.747],
    "GUBENG": [-7.276, 112.758],
    "KENJERAN": [-7.235, 112.800],
    "MULYOREJO": [-7.275, 112.785],
    "SIMOKERTO": [-7.245, 112.750],
    "TAMBAKSARI": [-7.255, 112.760],
}

peta = folium.Map(location=[-7.26, 112.77], zoom_start=12)

for _, row in pivot.iterrows():
    kec = row['kecamatan']
    
    if kec in koordinat:
        
        # Tentukan dominasi
        if row['PASANG BARU'] > row['TAMBAH DAYA']:
            warna = "blue"
            dominasi = "Pasang Baru"
        elif row['TAMBAH DAYA'] > row['PASANG BARU']:
            warna = "red"
            dominasi = "Tambah Daya"
        else:
            warna = "gray"
            dominasi = "Seimbang"
        
        folium.CircleMarker(
            location=koordinat[kec],
            radius=row['TOTAL']/200,
            color=warna,
            fill=True,
            fill_color=warna,
            fill_opacity=0.6,
            popup=f"""
            <b>{kec}</b><br><br>
            Pasang Baru: {row['PASANG BARU']} ({row['% PASANG BARU']}%)<br>
            Tambah Daya: {row['TAMBAH DAYA']} ({row['% TAMBAH DAYA']}%)<br><br>
            <b>Dominasi: {dominasi}</b><br>
            Total: {row['TOTAL']}
            """
        ).add_to(peta)

peta.save("peta_penyebaran vs tambah daya.html")
peta


# tabel rangking kecamatan 

# Pivot jumlah layanan
pivot = pd.crosstab(
    df_fokus['kecamatan'],
    df_fokus['jenis_transaksi']
).reset_index()

# Pastikan kolom ada
if 'PASANG BARU' not in pivot.columns:
    pivot['PASANG BARU'] = 0
if 'TAMBAH DAYA' not in pivot.columns:
    pivot['TAMBAH DAYA'] = 0

# Hitung total
pivot['TOTAL'] = pivot['PASANG BARU'] + pivot['TAMBAH DAYA']

# Ranking
pivot['RANK_TOTAL'] = pivot['TOTAL'].rank(ascending=False).astype(int)
pivot['RANK_PASANG_BARU'] = pivot['PASANG BARU'].rank(ascending=False).astype(int)
pivot['RANK_TAMBAH_DAYA'] = pivot['TAMBAH DAYA'].rank(ascending=False).astype(int)

# Urutkan berdasarkan total
ranking = pivot.sort_values('RANK_TOTAL')

ranking

import matplotlib.pyplot as plt

ranking_sorted = ranking.sort_values('TOTAL')

plt.figure()
plt.barh(ranking_sorted['kecamatan'], ranking_sorted['TOTAL'])
plt.title("Ranking Total Permohonan per Kecamatan")
plt.xlabel("Total Permohonan")
plt.ylabel("Kecamatan")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

tren_kecamatan = (
    df_fokus.groupby(['bulan','kecamatan'])
    .size()
    .unstack()
)

plt.figure()
tren_kecamatan.plot()

plt.title("Tren Bulanan per Kecamatan")
plt.xlabel("Bulan")
plt.ylabel("Jumlah Permohonan")
plt.xticks(rotation=45)

# Atur legend
plt.legend(
    title="Kecamatan",
    loc="upper right",      # kanan atas
    fontsize=8,             # ukuran kecil
    title_fontsize=9,
    frameon=True
)

plt.tight_layout()
plt.show()


# Growth kecamatan 
import pandas as pd
import matplotlib.pyplot as plt

# Pastikan kolom tanggal sudah datetime
df_fokus['tanggal_permohonan'] = pd.to_datetime(df_fokus['tanggal_permohonan'])
df_fokus['bulan'] = df_fokus['tanggal_permohonan'].dt.to_period('M')

# Agregasi bulanan per layanan
tren_bulanan = (
    df_fokus.groupby(['bulan','jenis_transaksi'])
    .size()
    .unstack()
)

# Hitung growth rate (%)
growth_bulanan = tren_bulanan.pct_change() * 100

# Plot growth rate
plt.figure()
growth_bulanan.plot()
plt.title("Growth Rate Bulanan (%)")
plt.xlabel("Bulan")
plt.ylabel("Persentase Perubahan")
plt.xticks(rotation=45)

plt.legend(
    title="Jenis Layanan",
    loc="upper right",
    fontsize=8
)

plt.tight_layout()
plt.show()

# kecamatan dengan pertumbuhan tercepat 

# Total bulanan per kecamatan
tren_kecamatan = (
    df_fokus.groupby(['bulan','kecamatan'])
    .size()
    .unstack()
)

# Hitung growth rate per kecamatan
growth_kecamatan = tren_kecamatan.pct_change() * 100

# Hitung rata-rata growth tiap kecamatan
rata_growth = growth_kecamatan.mean().sort_values(ascending=False)

rata_growth

plt.figure()
rata_growth.sort_values().plot(kind='barh')
plt.title("Rata-rata Growth Rate per Kecamatan (%)")
plt.xlabel("Rata-rata Pertumbuhan (%)")
plt.ylabel("Kecamatan")
plt.tight_layout()
plt.show()

# perbandingan pemasangan VS tambah daya 

# Rata-rata growth tiap layanan
rata_growth_layanan = growth_bulanan.mean().sort_values(ascending=False)

rata_growth_layanan

plt.figure()
rata_growth_layanan.sort_values().plot(kind='barh')
plt.title("Rata-rata Growth Rate per Jenis Layanan (%)")
plt.xlabel("Rata-rata Pertumbuhan (%)")
plt.ylabel("Jenis Layanan")
plt.tight_layout()
plt.show()

# KLASIFIKASI BEDASARKAN WILAYA
import pandas as pd
import matplotlib.pyplot as plt


# Q1 → Low
# Q2–Q3 → Medium
# Q4 → High

# Hitung total permohonan per kecamatan
total_kecamatan = (
    df_fokus.groupby('kecamatan')
    .size()
    .reset_index(name='TOTAL')
)

# Hitung kuartil
Q1 = total_kecamatan['TOTAL'].quantile(0.25)
Q3 = total_kecamatan['TOTAL'].quantile(0.75)

# Fungsi klasifikasi
def klasifikasi(total):
    if total <= Q1:
        return "Low Demand"
    elif total >= Q3:
        return "High Demand"
    else:
        return "Medium Demand"

total_kecamatan['Kategori'] = total_kecamatan['TOTAL'].apply(klasifikasi)

# Urutkan dari tertinggi
total_kecamatan = total_kecamatan.sort_values('TOTAL', ascending=False)

total_kecamatan

plt.figure()
plt.barh(total_kecamatan['kecamatan'], total_kecamatan['TOTAL'])
plt.title("Klasifikasi Permohonan per Kecamatan")
plt.xlabel("Total Permohonan")
plt.ylabel("Kecamatan")
plt.tight_layout()
plt.show()

# peta FIKS 

import pandas as pd
import folium

# Bersihkan format layanan
df_fokus['jenis_transaksi'] = df_fokus['jenis_transaksi'].str.upper().str.strip()

# Pivot jumlah layanan per kecamatan
pivot = pd.crosstab(
    df_fokus['kecamatan'],
    df_fokus['jenis_transaksi']
).reset_index()

# Pastikan kolom ada
if 'PASANG BARU' not in pivot.columns:
    pivot['PASANG BARU'] = 0
if 'TAMBAH DAYA' not in pivot.columns:
    pivot['TAMBAH DAYA'] = 0

pivot['TOTAL'] = pivot['PASANG BARU'] + pivot['TAMBAH DAYA']

# Hitung kuartil
Q1 = pivot['TOTAL'].quantile(0.25)
Q3 = pivot['TOTAL'].quantile(0.75)

def klasifikasi(total):
    if total <= Q1:
        return "Low Demand"
    elif total >= Q3:
        return "High Demand"
    else:
        return "Medium Demand"

pivot['Kategori'] = pivot['TOTAL'].apply(klasifikasi)

koordinat = {
    "GENTENG": [-7.265, 112.747],
    "GUBENG": [-7.276, 112.758],
    "KENJERAN": [-7.235, 112.800],
    "MULYOREJO": [-7.275, 112.785],
    "SIMOKERTO": [-7.245, 112.750],
    "TAMBAKSARI": [-7.255, 112.760],
}

# Buat peta dasar
peta = folium.Map(location=[-7.26, 112.77], zoom_start=12)

for _, row in pivot.iterrows():
    kec = row['kecamatan']
    
    if kec in koordinat:
        
        # Tentukan warna
        if row['Kategori'] == "High Demand":
            warna = "red"
        elif row['Kategori'] == "Medium Demand":
            warna = "orange"
        else:
            warna = "green"
        
        folium.CircleMarker(
            location=koordinat[kec],
            radius=row['TOTAL']/200,
            color=warna,
            fill=True,
            fill_color=warna,
            fill_opacity=0.7,
            popup=f"""
            <b>{kec}</b><br><br>
            Pasang Baru: {row['PASANG BARU']}<br>
            Tambah Daya: {row['TAMBAH DAYA']}<br>
            Total: {row['TOTAL']}<br><br>
            <b>Kategori: {row['Kategori']}</b>
            """
        ).add_to(peta)

legend_html = """
<div style="
position: fixed; 
top: 20px; right: 20px; width: 150px; 
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:12px;
padding: 10px;
">
<b>Kategori Demand</b><br>
<i style="color:red;">●</i> High Demand<br>
<i style="color:orange;">●</i> Medium Demand<br>
<i style="color:green;">●</i> Low Demand
</div>
"""

peta.get_root().html.add_child(folium.Element(legend_html))

peta.save("peta_pln penyebaran vs tambah daya.html")
peta

# INTEGRITAS SPASIAL + TEMPORAL
# siapkan data bulanan 

import pandas as pd

# Pastikan format datetime
df_fokus['tanggal_permohonan'] = pd.to_datetime(df_fokus['tanggal_permohonan'])

# Buat kolom bulan
df_fokus['bulan'] = df_fokus['tanggal_permohonan'].dt.to_period('M')

# hitung total per kecamatan

# Hitung total layanan per kecamatan per bulan
spatio_temporal = df_fokus.groupby(['bulan', 'kecamatan']).size().reset_index(name='TOTAL')

spatio_temporal.head()

# klasifikasi hight,medium,low per bulan 

def klasifikasi_bulanan(df):
    Q1 = df['TOTAL'].quantile(0.25)
    Q3 = df['TOTAL'].quantile(0.75)
    
    def kategori(x):
        if x <= Q1:
            return "Low"
        elif x >= Q3:
            return "High"
        else:
            return "Medium"
    
    df['Kategori'] = df['TOTAL'].apply(kategori)
    return df

spatio_temporal = spatio_temporal.groupby('bulan').apply(klasifikasi_bulanan)
spatio_temporal.reset_index(drop=True, inplace=True)

spatio_temporal.head()

# perubhan kategori per kecamatan

pivot_kategori = spatio_temporal.pivot(
    index='kecamatan',
    columns='bulan',
    values='Kategori'
)

pivot_kategori

# Kecamatan Konsisten High Demand
konsisten_high = pivot_kategori[
    pivot_kategori.apply(lambda row: (row == "High").sum(), axis=1) >= len(pivot_kategori.columns)*0.6
]

konsisten_high

# kecamatan yang naik dari medium ke hight

transisi = []

for kec in pivot_kategori.index:
    kategori_list = pivot_kategori.loc[kec].values
    
    for i in range(len(kategori_list)-1):
        if kategori_list[i] == "Medium" and kategori_list[i+1] == "High":
            transisi.append(kec)

kecamatan_naik = list(set(transisi))

kecamatan_naik

# heatmap 

import matplotlib.pyplot as plt
import numpy as np

# Copy pivot kategori
heatmap_data = pivot_kategori.copy()

# Konversi kategori ke angka
mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

heatmap_data = heatmap_data.replace(mapping)

heatmap_data

plt.figure()

plt.imshow(heatmap_data, aspect='auto')

plt.xticks(
    ticks=np.arange(len(heatmap_data.columns)),
    labels=heatmap_data.columns.astype(str),
    rotation=45
)

plt.yticks(
    ticks=np.arange(len(heatmap_data.index)),
    labels=heatmap_data.index
)

plt.colorbar(label="Kategori (1=Low, 2=Medium, 3=High)")

plt.title("Heatmap Perubahan Kategori Demand per Kecamatan")

plt.xlabel("Bulan")
plt.ylabel("Kecamatan")

plt.tight_layout()
plt.show()
