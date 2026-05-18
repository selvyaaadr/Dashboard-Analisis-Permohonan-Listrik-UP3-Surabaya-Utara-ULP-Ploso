import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard PBPD PLN",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Dashboard Analisis PBPD PLN 2025")
st.markdown("Analisis Permohonan Pasang Baru & Perubahan Daya")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    # CLEANING
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        'tglmohon': 'tanggal_permohonan',
        'idpel': 'id_pelanggan',
        'nama': 'nama_pelanggan',
        'alamat': 'alamat_pelanggan',
        'tarif': 'tarif_baru',
        'daya': 'daya_baru'
    })

    df['tanggal_permohonan'] = pd.to_datetime(df['tanggal_permohonan'], errors='coerce')
    df['daya_lama'] = pd.to_numeric(df.get('daya_lama', 0), errors='coerce').fillna(0)
    df['daya_baru'] = pd.to_numeric(df['daya_baru'], errors='coerce')
    df['selisih_daya'] = df['daya_baru'] - df['daya_lama']

    df['tahun'] = df['tanggal_permohonan'].dt.year
    df['bulan'] = df['tanggal_permohonan'].dt.month
    df['nama_bulan'] = df['tanggal_permohonan'].dt.strftime('%B')

    # Ekstrak kecamatan
    df['kecamatan'] = (
        df['alamat_pelanggan']
        .str.split(',')
        .str[-3]
        .str.strip()
        .str.upper()
    )

    if 'jenis_transaksi' in df.columns:
        df['jenis_transaksi'] = df['jenis_transaksi'].str.upper().str.strip()

    return df


# ─────────────────────────────────────────────
# SIDEBAR - UPLOAD FILE
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Data")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    st.markdown("---")
    st.markdown("**Filter Data**")

if uploaded_file is None:
    st.info("⬅️ Silakan upload file CSV terlebih dahulu di sidebar kiri.")
    st.stop()

df = load_data(uploaded_file)

# Filter sidebar
with st.sidebar:
    tahun_tersedia = sorted(df['tahun'].dropna().unique())
    tahun_pilih = st.multiselect("Pilih Tahun", tahun_tersedia, default=tahun_tersedia)

    kecamatan_fokus = [
        "GENTENG", "GUBENG", "KENJERAN",
        "MULYOREJO", "SIMOKERTO", "TAMBAKSARI"
    ]
    kec_all = sorted(df['kecamatan'].dropna().unique())
    kec_pilih = st.multiselect("Pilih Kecamatan", kec_all,
                                default=[k for k in kecamatan_fokus if k in kec_all])

df_filtered = df[df['tahun'].isin(tahun_pilih)] if tahun_pilih else df
df_fokus = df_filtered[df_filtered['kecamatan'].isin(kec_pilih)] if kec_pilih else df_filtered

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.subheader("📊 Ringkasan Data")
col1, col2, col3, col4 = st.columns(4)

total_permohonan = len(df_filtered)
total_pasang_baru = (df_filtered['jenis_transaksi'] == 'PASANG BARU').sum() if 'jenis_transaksi' in df_filtered.columns else 0
total_tambah_daya = df_filtered['jenis_transaksi'].str.upper().str.strip().str.contains('TAMBAH', na=False).sum()
total_pasang_baru = df_filtered['jenis_transaksi'].str.upper().str.strip().str.contains('PASANG', na=False).sum()
pct_naik = round((df_filtered['selisih_daya'] > 0).sum() / total_permohonan * 100, 1) if total_permohonan > 0 else 0

col1.metric("Total Permohonan", f"{total_permohonan:,}")
col2.metric("Pasang Baru", f"{total_pasang_baru:,}")
col3.metric("Tambah Daya", f"{total_tambah_daya:,}")
col4.metric("% Naik Daya", f"{pct_naik}%")

st.markdown("---")

# ─────────────────────────────────────────────
# TAB NAVIGASI
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Analisis Temporal",
    "🔄 Jenis Transaksi",
    "⚡ Perubahan Daya",
    "🗺️ Analisis Spasial",
    "📋 Data Mentah"
])

# ════════════════════════════════════════════
# TAB 1: ANALISIS TEMPORAL
# ════════════════════════════════════════════
with tab1:
    st.subheader("📅 Analisis Temporal Permohonan")

    # Tren Bulanan
    permohonan_bulanan = (
        df_filtered
        .groupby(pd.Grouper(key='tanggal_permohonan', freq='ME'))
        .size()
        .reset_index(name='jumlah_permohonan')
    )
    permohonan_bulanan['moving_avg'] = (
        permohonan_bulanan['jumlah_permohonan'].rolling(window=3).mean()
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(permohonan_bulanan['tanggal_permohonan'],
                permohonan_bulanan['jumlah_permohonan'], label='Aktual')
        ax.plot(permohonan_bulanan['tanggal_permohonan'],
                permohonan_bulanan['moving_avg'], linestyle='--', label='Moving Avg (3bln)')
        ax.set_title("Tren Permohonan + Moving Average")
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Jumlah Permohonan")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        # Growth Rate
        permohonan_bln = (
            df_filtered.groupby('bulan').size()
            .reset_index(name='jumlah').sort_values('bulan')
        )
        permohonan_bln['growth_%'] = permohonan_bln['jumlah'].pct_change() * 100

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(permohonan_bln['bulan'], permohonan_bln['growth_%'], marker='o', color='orange')
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax.set_title("Growth Rate Bulanan (%)")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Pertumbuhan (%)")
        ax.set_xticks(range(1, 13))
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Distribusi per Bulan
    top_bulan = df_filtered.groupby('nama_bulan').size().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    top_bulan.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title("Distribusi Permohonan per Bulan")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Highlight tertinggi & terendah
    pb_monthly = (
        df_filtered.groupby('bulan').size()
        .reset_index(name='jumlah_permohonan').sort_values('bulan')
    )
    if not pb_monthly.empty:
        idx_max = pb_monthly['jumlah_permohonan'].idxmax()
        idx_min = pb_monthly['jumlah_permohonan'].idxmin()
        c1, c2 = st.columns(2)
        c1.success(f"🔺 Bulan Tertinggi: **Bulan {int(pb_monthly.loc[idx_max,'bulan'])}** "
                   f"({int(pb_monthly.loc[idx_max,'jumlah_permohonan'])} permohonan)")
        c2.error(f"🔻 Bulan Terendah: **Bulan {int(pb_monthly.loc[idx_min,'bulan'])}** "
                 f"({int(pb_monthly.loc[idx_min,'jumlah_permohonan'])} permohonan)")

# ════════════════════════════════════════════
# TAB 2: JENIS TRANSAKSI
# ════════════════════════════════════════════
with tab2:
    st.subheader("🔄 Analisis Jenis Transaksi")

    if 'jenis_transaksi' not in df_filtered.columns:
        st.warning("Kolom 'jenis_transaksi' tidak ditemukan.")
    else:
        transaksi_counts = df_filtered['jenis_transaksi'].value_counts().reset_index()
        transaksi_counts.columns = ['jenis_transaksi', 'jumlah']
        transaksi_counts['persentase (%)'] = (
            transaksi_counts['jumlah'] / transaksi_counts['jumlah'].sum() * 100
        ).round(2)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(transaksi_counts['jenis_transaksi'], transaksi_counts['jumlah'], color=['steelblue','coral'])
            ax.set_title("Distribusi Jenis Transaksi")
            ax.set_xlabel("Jenis Transaksi")
            ax.set_ylabel("Jumlah")
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(transaksi_counts['jumlah'],
                   labels=transaksi_counts['jenis_transaksi'],
                   autopct='%1.1f%%',
                   colors=['steelblue', 'coral', 'gold', 'lightgreen'])
            ax.set_title("Persentase Jenis Transaksi")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.dataframe(transaksi_counts, use_container_width=True)

        # Tren per Tahun
        transaksi_tahunan = (
            df_filtered.groupby(['tahun', 'jenis_transaksi'])
            .size().unstack().fillna(0)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        transaksi_tahunan.plot(kind='bar', ax=ax)
        ax.set_title("Distribusi Jenis Transaksi per Tahun")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Jumlah")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Tren Bulanan per Transaksi
        transaksi_bulanan = (
            df_filtered.groupby(['bulan', 'jenis_transaksi'])
            .size().unstack().fillna(0)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        for kolom in transaksi_bulanan.columns:
            ax.plot(transaksi_bulanan.index, transaksi_bulanan[kolom], marker='o', label=kolom)
        ax.set_title("Tren Jenis Transaksi per Bulan")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Jumlah")
        ax.set_xticks(range(1, 13))
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Heatmap Bulanan vs Transaksi
        heatmap_data = (
            df_filtered.groupby(['bulan', 'jenis_transaksi'])
            .size().unstack().fillna(0).sort_index()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(heatmap_data.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(np.arange(len(heatmap_data.columns)))
        ax.set_xticklabels(heatmap_data.columns, rotation=30)
        ax.set_yticks(np.arange(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index)
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                ax.text(j, i, int(heatmap_data.values[i, j]),
                        ha='center', va='center', fontsize=9)
        fig.colorbar(im, label='Jumlah Permohonan')
        ax.set_title("Heatmap Bulanan vs Jenis Transaksi")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════
# TAB 3: PERUBAHAN DAYA
# ════════════════════════════════════════════
with tab3:
    st.subheader("⚡ Analisis Perubahan Daya")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Statistik Selisih Daya:**")
        st.dataframe(df_filtered['selisih_daya'].describe().to_frame(), use_container_width=True)

    with col2:
        menaikkan = (df_filtered['selisih_daya'] > 0).sum()
        total = len(df_filtered)
        st.metric("Pelanggan Menaikkan Daya", f"{menaikkan:,}",
                  delta=f"{round(menaikkan/total*100,2)}% dari total" if total > 0 else "")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_filtered['selisih_daya'].dropna(), bins=20, color='steelblue', edgecolor='white')
    ax.set_title("Distribusi Selisih Daya")
    ax.set_xlabel("Selisih Daya (VA)")
    ax.set_ylabel("Jumlah Pelanggan")
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Rata-rata per Bulan
    rata_bulanan = (
        df_filtered.groupby('bulan')['selisih_daya']
        .mean().reset_index().sort_values('bulan')
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rata_bulanan['bulan'], rata_bulanan['selisih_daya'], marker='o', color='green')
    ax.set_title("Rata-rata Kenaikan Daya per Bulan")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Rata-rata Selisih Daya (VA)")
    ax.set_xticks(range(1, 13))
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════
# TAB 4: ANALISIS SPASIAL
# ════════════════════════════════════════════
with tab4:
    st.subheader("🗺️ Analisis Spasial per Kecamatan")

    # Total permohonan per kecamatan
    total_kecamatan = (
        df_fokus.groupby('kecamatan').size()
        .reset_index(name='TOTAL')
    )

    if total_kecamatan.empty:
        st.warning("Tidak ada data untuk kecamatan yang dipilih.")
    else:
        # Klasifikasi demand
        Q1 = total_kecamatan['TOTAL'].quantile(0.25)
        Q3 = total_kecamatan['TOTAL'].quantile(0.75)

        def klasifikasi(total):
            if total <= Q1:
                return "Low Demand"
            elif total >= Q3:
                return "High Demand"
            else:
                return "Medium Demand"

        total_kecamatan['Kategori'] = total_kecamatan['TOTAL'].apply(klasifikasi)
        total_kecamatan = total_kecamatan.sort_values('TOTAL', ascending=False)

        col1, col2 = st.columns([2, 1])

        with col1:
            color_map = {'High Demand': 'red', 'Medium Demand': 'orange', 'Low Demand': 'green'}
            colors = total_kecamatan['Kategori'].map(color_map)

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(total_kecamatan['kecamatan'], total_kecamatan['TOTAL'],
                           color=colors)
            ax.set_title("Klasifikasi Permohonan per Kecamatan")
            ax.set_xlabel("Total Permohonan")

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', label='High Demand'),
                               Patch(facecolor='orange', label='Medium Demand'),
                               Patch(facecolor='green', label='Low Demand')]
            ax.legend(handles=legend_elements, loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.write("**Tabel Klasifikasi:**")
            st.dataframe(total_kecamatan.reset_index(drop=True), use_container_width=True)

        # Growth Rate per Kecamatan
        if 'jenis_transaksi' in df_fokus.columns:
            df_fokus2 = df_fokus.copy()
            df_fokus2['bulan_period'] = df_fokus2['tanggal_permohonan'].dt.to_period('M')

            tren_kecamatan = (
                df_fokus2.groupby(['bulan_period', 'kecamatan'])
                .size().unstack().fillna(0)
            )
            growth_kecamatan = tren_kecamatan.pct_change() * 100
            rata_growth = growth_kecamatan.mean().sort_values()

            fig, ax = plt.subplots(figsize=(8, 4))
            rata_growth.plot(kind='barh', ax=ax, color='teal')
            ax.set_title("Rata-rata Growth Rate per Kecamatan (%)")
            ax.set_xlabel("Rata-rata Pertumbuhan (%)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── PETA INTERAKTIF FOLIUM ──
        st.markdown("### 🗺️ Peta Interaktif Sebaran Permohonan")

        koordinat = {
            "GENTENG":    [-7.265, 112.747],
            "GUBENG":     [-7.276, 112.758],
            "KENJERAN":   [-7.235, 112.800],
            "MULYOREJO":  [-7.275, 112.785],
            "SIMOKERTO":  [-7.245, 112.750],
            "TAMBAKSARI": [-7.255, 112.760],
        }

        # Pivot data untuk peta
        if 'jenis_transaksi' in df_fokus.columns:
            pivot_peta = pd.crosstab(df_fokus['kecamatan'], df_fokus['jenis_transaksi']).reset_index()
        else:
            pivot_peta = total_kecamatan[['kecamatan']].copy()

        pivot_peta = pivot_peta.merge(total_kecamatan[['kecamatan','TOTAL','Kategori']], on='kecamatan', how='left')

        peta = folium.Map(location=[-7.26, 112.77], zoom_start=13)

        for _, row in pivot_peta.iterrows():
            kec = row['kecamatan']
            if kec not in koordinat:
                continue

            warna = {"High Demand": "red", "Medium Demand": "orange", "Low Demand": "green"}.get(row.get('Kategori',''), 'blue')
            total_val = int(row.get('TOTAL', 0))
            radius = max(8, total_val / 150)

            pb = int(row.get('PASANG BARU', 0)) if 'PASANG BARU' in row else '-'
            td = int(row.get('TAMBAH DAYA', 0)) if 'TAMBAH DAYA' in row else '-'

            popup_html = f"""
            <div style='font-family:Arial; font-size:13px; min-width:160px'>
                <b style='font-size:15px'>📍 {kec}</b><br><br>
                🔌 Pasang Baru : <b>{pb}</b><br>
                ⚡ Tambah Daya : <b>{td}</b><br>
                📊 Total       : <b>{total_val}</b><br><br>
                <span style='background:{warna};color:white;padding:2px 8px;border-radius:4px'>
                    {row.get('Kategori','')}
                </span>
            </div>
            """

            folium.CircleMarker(
                location=koordinat[kec],
                radius=radius,
                color=warna,
                fill=True,
                fill_color=warna,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{kec} — {total_val} permohonan"
            ).add_to(peta)

        # Legend
        legend_html = """
        <div style="
            position: fixed; top: 20px; right: 20px; width: 160px;
            background-color: white; border: 2px solid grey;
            z-index: 9999; font-size: 12px; padding: 10px; border-radius: 6px;">
            <b>🗺️ Kategori Demand</b><br><br>
            <i style='color:red;font-size:16px'>●</i> High Demand<br>
            <i style='color:orange;font-size:16px'>●</i> Medium Demand<br>
            <i style='color:green;font-size:16px'>●</i> Low Demand<br>
            <br><small>Ukuran lingkaran = total permohonan</small>
        </div>
        """
        peta.get_root().html.add_child(folium.Element(legend_html))

        st_folium(peta, width=900, height=480)

        # ── HEATMAP SPASIAL-TEMPORAL ──
        if 'jenis_transaksi' in df_fokus.columns:
            df_fokus3 = df_fokus.copy()
            df_fokus3['bulan_period'] = df_fokus3['tanggal_permohonan'].dt.to_period('M')

            spatio_temporal = df_fokus3.groupby(['bulan_period', 'kecamatan']).size().reset_index(name='TOTAL')

            def klasifikasi_bulanan(group):
                Q1b = group['TOTAL'].quantile(0.25)
                Q3b = group['TOTAL'].quantile(0.75)
                def kat(x):
                    if x <= Q1b: return 1
                    elif x >= Q3b: return 3
                    else: return 2
                group['val'] = group['TOTAL'].apply(kat)
                return group

            st_result = spatio_temporal.groupby('bulan_period', group_keys=False).apply(klasifikasi_bulanan)
            pivot_ht = st_result.pivot(index='kecamatan', columns='bulan_period', values='val')

            if not pivot_ht.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                im = ax.imshow(pivot_ht.values, aspect='auto', cmap='RdYlGn')
                ax.set_xticks(np.arange(len(pivot_ht.columns)))
                ax.set_xticklabels(pivot_ht.columns.astype(str), rotation=45)
                ax.set_yticks(np.arange(len(pivot_ht.index)))
                ax.set_yticklabels(pivot_ht.index)
                fig.colorbar(im, label="1=Low  2=Medium  3=High")
                ax.set_title("Heatmap Perubahan Kategori Demand per Kecamatan")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ════════════════════════════════════════════
# TAB 5: DATA MENTAH
# ════════════════════════════════════════════
with tab5:
    st.subheader("📋 Data Mentah")
    st.write(f"Total baris: **{len(df_filtered):,}**")

    search = st.text_input("🔍 Cari nama pelanggan atau kecamatan:")
    display_df = df_filtered
    if search:
        mask = (
            df_filtered.get('nama_pelanggan', pd.Series(dtype=str)).str.contains(search, case=False, na=False) |
            df_filtered.get('kecamatan', pd.Series(dtype=str)).str.contains(search, case=False, na=False)
        )
        display_df = df_filtered[mask]

    st.dataframe(display_df.head(500), use_container_width=True)
    st.caption("Menampilkan maksimal 500 baris pertama.")

    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "data_filtered.csv", "text/csv")
