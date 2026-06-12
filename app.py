import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import networkx as nx
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard PLN – Permohonan Listrik",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Dashboard Analisis Permohonan Listrik")
st.markdown("**Sumber data:** PBPD 2025 MAGANG.csv")
st.divider()

# ─────────────────────────────────────────
# UPLOAD / LOAD DATA
# ─────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader(
    "📂 Upload file CSV", type=["csv"]
)

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    # Bersihkan nama kolom
    df.columns = df.columns.str.strip().str.lower()

    # Rename kolom utama
    rename_map = {
        'tglmohon':   'tanggal_permohonan',
        'idpel':      'id_pelanggan',
        'nama':       'nama_pelanggan',
        'alamat':     'alamat_pelanggan',
        'tarif':      'tarif_baru',
        'daya':       'daya_baru',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Format tanggal
    df['tanggal_permohonan'] = pd.to_datetime(df['tanggal_permohonan'], errors='coerce')

    # Numerik
    df['daya_lama'] = pd.to_numeric(df.get('daya_lama', 0), errors='coerce').fillna(0)
    df['daya_baru'] = pd.to_numeric(df['daya_baru'], errors='coerce')

    # Selisih daya
    df['selisih_daya'] = df['daya_baru'] - df['daya_lama']

    # Waktu
    df['tahun']      = df['tanggal_permohonan'].dt.year
    df['bulan']      = df['tanggal_permohonan'].dt.month
    df['nama_bulan'] = df['tanggal_permohonan'].dt.strftime('%B')

    # Kecamatan
    if 'alamat_pelanggan' in df.columns:
        df['kecamatan'] = (
            df['alamat_pelanggan']
            .str.split(',')
            .str[-3]
            .str.upper()
            .str.strip()
        )

    # Jenis transaksi – uppercase & normalisasi nama
    if 'jenis_transaksi' in df.columns:
        df['jenis_transaksi'] = df['jenis_transaksi'].str.upper().str.strip()
        df['jenis_transaksi'] = df['jenis_transaksi'].replace({
            'PERUBAHAN DAYA': 'TAMBAH DAYA',
            'UBAH DAYA':      'TAMBAH DAYA',
            'NAIK DAYA':      'TAMBAH DAYA',
        })

    return df


# ─────────────────────────────────────────
# GUARD – belum ada file
# ─────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Silakan upload file **PBPD 2025 MAGANG.csv** melalui sidebar untuk memulai.")
    st.stop()

df = load_data(uploaded_file)

# ─────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────
KECAMATAN_FOKUS = [
    "GENTENG", "GUBENG", "KENJERAN",
    "MULYOREJO", "SIMOKERTO", "TAMBAKSARI",
]

KOORDINAT = {
    "GENTENG":    [-7.265, 112.747],
    "GUBENG":     [-7.276, 112.758],
    "KENJERAN":   [-7.235, 112.800],
    "MULYOREJO":  [-7.275, 112.785],
    "SIMOKERTO":  [-7.245, 112.750],
    "TAMBAKSARI": [-7.255, 112.760],
}

MAPPING_BULAN = {
    1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
    7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'
}

df_fokus = df[df['kecamatan'].isin(KECAMATAN_FOKUS)].copy()

# ─────────────────────────────────────────
# SIDEBAR FILTER
# ─────────────────────────────────────────
st.sidebar.header("🔍 Filter")

tahun_list = sorted(df['tahun'].dropna().unique().astype(int))
tahun_sel  = st.sidebar.multiselect("Tahun", tahun_list, default=tahun_list)

bulan_list = list(range(1, 13))
bulan_sel  = st.sidebar.multiselect(
    "Bulan", bulan_list,
    default=bulan_list,
    format_func=lambda x: MAPPING_BULAN[x]
)

kec_sel = st.sidebar.multiselect(
    "Kecamatan (fokus)", KECAMATAN_FOKUS, default=KECAMATAN_FOKUS
)

# Terapkan filter
df_f       = df[(df['tahun'].isin(tahun_sel)) & (df['bulan'].isin(bulan_sel))]
df_fok_f   = df_fokus[
    (df_fokus['tahun'].isin(tahun_sel)) &
    (df_fokus['bulan'].isin(bulan_sel)) &
    (df_fokus['kecamatan'].isin(kec_sel))
]

# ─────────────────────────────────────────
# METRIK RINGKASAN
# ─────────────────────────────────────────
st.subheader("📊 Ringkasan")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Permohonan",  f"{len(df_f):,}")
col2.metric("Jenis Transaksi",   df_f['jenis_transaksi'].nunique() if 'jenis_transaksi' in df_f.columns else "–")
col3.metric("Rata-rata Selisih Daya (VA)", f"{df_f['selisih_daya'].mean():,.0f}")
col4.metric("Kecamatan Terdampak", df_fok_f['kecamatan'].nunique())

st.divider()

# ═══════════════════════════════════════════
# TAB NAVIGASI
# ═══════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📅 Analisis Temporal",
    "🔄 Jenis Transaksi",
    "📍 Analisis Spasial",
    "🗺️ Peta Interaktif",
    "🤖 Clustering",
    "📋 Gantt Chart",
    "📐 Teknik Optimasi",
    "🕸️ Graph Sains Data",
])

# ───────────────────────────────────────────
# TAB 1 – ANALISIS TEMPORAL
# ───────────────────────────────────────────
with tab1:
    st.header("Analisis Temporal")

    # Permohonan bulanan
    permohonan_bulanan = (
        df_f.groupby('bulan')
        .size()
        .reset_index(name='jumlah_permohonan')
        .sort_values('bulan')
    )
    permohonan_bulanan['moving_avg'] = (
        permohonan_bulanan['jumlah_permohonan']
        .rolling(window=3, min_periods=1)
        .mean()
    )
    permohonan_bulanan['growth_%'] = (
        permohonan_bulanan['jumlah_permohonan']
        .pct_change() * 100
    )
    permohonan_bulanan['nama_bulan'] = (
        permohonan_bulanan['bulan'].map(MAPPING_BULAN)
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Tren Permohonan + Moving Average (3 bln)")
        fig, ax = plt.subplots()
        ax.plot(permohonan_bulanan['bulan'],
                permohonan_bulanan['jumlah_permohonan'],
                marker='o', label='Aktual')
        ax.plot(permohonan_bulanan['bulan'],
                permohonan_bulanan['moving_avg'],
                linestyle='--', label='Moving Avg')
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Jumlah Permohonan")
        ax.set_xticks(range(1, 13))
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    with c2:
        st.subheader("Growth Rate Bulanan (%)")
        fig2, ax2 = plt.subplots()
        ax2.plot(permohonan_bulanan['bulan'],
                 permohonan_bulanan['growth_%'],
                 marker='o', color='tomato')
        ax2.axhline(0, color='gray', linewidth=0.8)
        ax2.set_xlabel("Bulan")
        ax2.set_ylabel("Pertumbuhan (%)")
        ax2.set_xticks(range(1, 13))
        ax2.grid(True)
        st.pyplot(fig2)
        plt.close()

    st.subheader("Distribusi Permohonan per Bulan (Bar)")
    top_bulan = (
        df_f.groupby('nama_bulan')
        .size()
        .reset_index(name='jumlah')
    )
    fig3 = px.bar(top_bulan, x='nama_bulan', y='jumlah',
                  labels={'nama_bulan': 'Bulan', 'jumlah': 'Jumlah'},
                  color='jumlah', color_continuous_scale='Blues')
    st.plotly_chart(fig3, use_container_width=True)

    # Highlight
    if not permohonan_bulanan.empty:
        bln_max = permohonan_bulanan.loc[permohonan_bulanan['jumlah_permohonan'].idxmax()]
        bln_min = permohonan_bulanan.loc[permohonan_bulanan['jumlah_permohonan'].idxmin()]
        cc1, cc2 = st.columns(2)
        cc1.success(
            f"📈 **Bulan Tertinggi:** {MAPPING_BULAN.get(int(bln_max['bulan']), '?')} "
            f"– {int(bln_max['jumlah_permohonan']):,} permohonan"
        )
        cc2.error(
            f"📉 **Bulan Terendah:** {MAPPING_BULAN.get(int(bln_min['bulan']), '?')} "
            f"– {int(bln_min['jumlah_permohonan']):,} permohonan"
        )

    # ═══════════════════════════════════════════════════════════════════
    # PEMBELAJARAN MESIN LANJUT – DEEP LEARNING (sesuai RPS UNESA 2024)
    # Materi: ANN/MLP (Minggu 1-5), Regularisasi (Minggu 6-7),
    #         Representation Learning / Autoencoder (Minggu 9-11)
    # ═══════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🧠 Deep Learning – Analisis Lanjut Data Temporal")

    dl_tab1, dl_tab2, dl_tab3 = st.tabs([
        "⚡ MLP Forecasting",
        "🔧 Regularisasi & Training Curve",
        "🔍 Anomaly Detection (Autoencoder)",
    ])

    # ── Persiapan data bersama ────────────────────────────────────────
    pb_monthly = (
        df_f.groupby('bulan')
        .size()
        .reset_index(name='jumlah')
        .sort_values('bulan')
    )

    # ─────────────────────────────────────────────────────────────────
    # DL-TAB 1 · MLP Forecasting (ANN/DNN – Minggu 1-5)
    # Konsep: Multi-Layer Perceptron, Backpropagation, Forward pass
    # ─────────────────────────────────────────────────────────────────
    with dl_tab1:
        if len(pb_monthly) < 5:
            st.warning("Data tidak cukup untuk melatih MLP (minimal 5 bulan).")
        else:
            # ── Feature engineering ──────────────────────────────────
            df_dl = pb_monthly.copy()
            df_dl['lag_1']   = df_dl['jumlah'].shift(1)
            df_dl['lag_2']   = df_dl['jumlah'].shift(2)
            df_dl['mav_3']   = df_dl['jumlah'].rolling(3, min_periods=1).mean().shift(1)
            df_dl = df_dl.dropna()

            X_dl = df_dl[['bulan', 'lag_1', 'lag_2', 'mav_3']].values
            y_dl = df_dl['jumlah'].values

            # Normalisasi (MinMaxScaler) – penting untuk ANN
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_scaled_dl = scaler_X.fit_transform(X_dl)
            y_scaled_dl = scaler_y.fit_transform(y_dl.reshape(-1, 1)).ravel()

            # ── Konfigurasi arsitektur MLP ────────────────────────────
            st.markdown("##### ⚙️ Konfigurasi Arsitektur MLP")
            col_arch1, col_arch2, col_arch3 = st.columns(3)
            with col_arch1:
                hidden_1 = st.selectbox("Hidden Layer 1 (neuron)", [16, 32, 64, 128], index=1)
            with col_arch2:
                hidden_2 = st.selectbox("Hidden Layer 2 (neuron)", [0, 8, 16, 32], index=2)
            with col_arch3:
                max_iter_mlp = st.slider("Max Iterasi (Epoch)", 100, 1000, 300, step=50)

            hidden_layers = (hidden_1,) if hidden_2 == 0 else (hidden_1, hidden_2)

            # ── Training MLP ──────────────────────────────────────────
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                max_iter=max_iter_mlp,
                random_state=42,
                early_stopping=False,
                learning_rate_init=0.001,
            )
            mlp.fit(X_scaled_dl, y_scaled_dl)

            y_pred_scaled = mlp.predict(X_scaled_dl)
            y_pred_dl = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_actual  = y_dl

            rmse_mlp = np.sqrt(mean_squared_error(y_actual, y_pred_dl))
            mae_mlp  = mean_absolute_error(y_actual, y_pred_dl)

            # ── Prediksi n bulan ke depan ─────────────────────────────
            st.markdown("##### 📅 Prediksi Bulan ke Depan")
            n_forecast = st.slider("Jumlah bulan prediksi", 1, 6, 3)

            forecast_vals = list(y_dl)
            last_bulan    = int(df_dl['bulan'].iloc[-1])

            for i in range(n_forecast):
                b       = (last_bulan + i) % 12 + 1
                lag1    = forecast_vals[-1]
                lag2    = forecast_vals[-2]
                mav     = np.mean(forecast_vals[-3:])
                x_new   = scaler_X.transform([[b, lag1, lag2, mav]])
                y_new   = scaler_y.inverse_transform(mlp.predict(x_new).reshape(-1, 1))[0][0]
                forecast_vals.append(max(0, y_new))

            forecast_only = forecast_vals[len(y_dl):]
            bulan_fc      = [(last_bulan + i) % 12 + 1 for i in range(1, n_forecast + 1)]
            nama_fc       = [MAPPING_BULAN.get(b, str(b)) for b in bulan_fc]

            # ── Plot hasil ────────────────────────────────────────────
            fig_mlp = go.Figure()
            fig_mlp.add_trace(go.Scatter(
                x=df_dl['bulan'].tolist(),
                y=y_actual.tolist(),
                mode='lines+markers',
                name='Aktual',
                line=dict(color='royalblue', width=2),
            ))
            fig_mlp.add_trace(go.Scatter(
                x=df_dl['bulan'].tolist(),
                y=y_pred_dl.tolist(),
                mode='lines+markers',
                name='Prediksi MLP (train)',
                line=dict(color='orange', dash='dash', width=2),
            ))
            fig_mlp.add_trace(go.Scatter(
                x=bulan_fc,
                y=forecast_only,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='green', dash='dot', width=2),
                marker=dict(symbol='star', size=10),
            ))
            fig_mlp.update_layout(
                title=f"MLP Forecasting – Arsitektur {hidden_layers}",
                xaxis_title="Bulan",
                yaxis_title="Jumlah Permohonan",
                xaxis=dict(tickvals=list(range(1, 13)),
                           ticktext=list(MAPPING_BULAN.values())),
                legend=dict(orientation='h', y=-0.2),
            )
            st.plotly_chart(fig_mlp, use_container_width=True)

            # ── Metrik evaluasi ───────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE",  f"{rmse_mlp:.2f}")
            m2.metric("MAE",   f"{mae_mlp:.2f}")
            m3.metric("Iterasi Konvergen", mlp.n_iter_)

            # ── Tabel forecast ────────────────────────────────────────
            st.markdown("##### 📋 Tabel Hasil Forecast")
            df_fc_tbl = pd.DataFrame({
                "Bulan": nama_fc,
                "Prediksi Permohonan": [round(v) for v in forecast_only],
            })
            st.dataframe(df_fc_tbl, use_container_width=True)

            # ── Penjelasan arsitektur ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    # DL-TAB 2 · Regularisasi & Training Curve
    # ─────────────────────────────────────────────────────────────────
    with dl_tab2:
        if len(pb_monthly) < 5:
            st.warning("Data tidak cukup untuk eksperimen regularisasi.")
        else:
            df_reg = pb_monthly.copy()
            df_reg['lag_1'] = df_reg['jumlah'].shift(1)
            df_reg['lag_2'] = df_reg['jumlah'].shift(2)
            df_reg['mav_3'] = df_reg['jumlah'].rolling(3, min_periods=1).mean().shift(1)
            df_reg = df_reg.dropna()

            X_reg = df_reg[['bulan', 'lag_1', 'lag_2', 'mav_3']].values
            y_reg = df_reg['jumlah'].values
            sx    = MinMaxScaler(); sy = MinMaxScaler()
            Xr    = sx.fit_transform(X_reg)
            yr    = sy.fit_transform(y_reg.reshape(-1, 1)).ravel()

            # ── Slider regularisasi ───────────────────────────────────
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                alpha_val = st.select_slider(
                    "Alpha L2 Regularisasi",
                    options=[0.0, 0.0001, 0.001, 0.01, 0.1],
                    value=0.0001,
                )
            with col_r2:
                use_early = st.checkbox("Gunakan Early Stopping", value=True)
                val_frac  = st.slider("Validation Fraction", 0.1, 0.3, 0.2, 0.05,
                                      disabled=not use_early)

            # ── 3 Konfigurasi: No-reg, L2, L2+EarlyStopping ──────────
            configs = {
                "Tanpa Regularisasi": dict(
                    alpha=0.0, early_stopping=False, max_iter=500),
                f"L2 (α={alpha_val})": dict(
                    alpha=alpha_val, early_stopping=False, max_iter=500),
                f"L2 + Early Stopping": dict(
                    alpha=alpha_val, early_stopping=use_early,
                    validation_fraction=val_frac, max_iter=500),
            }

            fig_lc, ax_lc = plt.subplots(figsize=(10, 4))
            reg_results = []

            for label, params in configs.items():
                m_reg = MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation='relu',
                    solver='adam',
                    random_state=42,
                    learning_rate_init=0.001,
                    **params,
                )
                m_reg.fit(Xr, yr)
                loss_curve = m_reg.loss_curve_
                ax_lc.plot(loss_curve, label=f"{label} ({len(loss_curve)} iter)")

                y_p  = sy.inverse_transform(m_reg.predict(Xr).reshape(-1, 1)).ravel()
                rmse = np.sqrt(mean_squared_error(y_reg, y_p))
                mae  = mean_absolute_error(y_reg, y_p)
                reg_results.append({
                    "Konfigurasi": label,
                    "Iterasi": len(loss_curve),
                    "RMSE": round(rmse, 2),
                    "MAE":  round(mae, 2),
                    "Loss Akhir": round(loss_curve[-1], 6),
                })

            ax_lc.set_xlabel("Epoch (Iterasi)")
            ax_lc.set_ylabel("Loss (MSE)")
            ax_lc.set_title("Training Loss Curve – Perbandingan Regularisasi")
            ax_lc.legend()
            ax_lc.grid(True, alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig_lc)
            plt.close()

            # ── Tabel perbandingan ────────────────────────────────────
            st.subheader("📋 Perbandingan Hasil Regularisasi")
            df_reg_tbl = pd.DataFrame(reg_results)
            st.dataframe(df_reg_tbl, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────
    # DL-TAB 3 · Autoencoder Anomaly Detection
    # ─────────────────────────────────────────────────────────────────
    with dl_tab3:
        if len(pb_monthly) < 4:
            st.warning("Data tidak cukup untuk Autoencoder (minimal 4 bulan).")
        else:
            # ── Siapkan data ──────────────────────────────────────────
            df_ae = pb_monthly.copy()
            df_ae['lag_1']   = df_ae['jumlah'].shift(1).fillna(method='bfill')
            df_ae['lag_2']   = df_ae['jumlah'].shift(2).fillna(method='bfill')
            df_ae['mav_3']   = df_ae['jumlah'].rolling(3, min_periods=1).mean()
            df_ae['growth']  = df_ae['jumlah'].pct_change().fillna(0)

            features_ae = ['jumlah', 'lag_1', 'lag_2', 'mav_3', 'growth']
            X_ae_raw    = df_ae[features_ae].values
            scaler_ae   = MinMaxScaler()
            X_ae        = scaler_ae.fit_transform(X_ae_raw)

            # ── Slider threshold & bottleneck ─────────────────────────
            col_ae1, col_ae2 = st.columns(2)
            with col_ae1:
                bottleneck = st.selectbox("Ukuran Bottleneck (dimensi laten)", [1, 2, 3], index=1)
            with col_ae2:
                threshold_pct = st.slider(
                    "Threshold Anomali (persentil reconstruction error)", 50, 95, 75
                )

            # ── Simulasi Autoencoder dengan MLP (Encoder + Decoder) ───
            # Encoder: 5 → bottleneck, Decoder: bottleneck → 5
            n_feat = X_ae.shape[1]

            # Encoder
            enc = MLPRegressor(
                hidden_layer_sizes=(bottleneck,),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                learning_rate_init=0.001,
            )
            # Fit encoder (input → compressed)
            enc.fit(X_ae, X_ae[:, :bottleneck])
            X_encoded = enc.predict(X_ae)

            # Decoder (compressed → reconstructed)
            if bottleneck == 1:
                X_enc_in = X_encoded.reshape(-1, 1)
            else:
                X_enc_in = np.column_stack([X_encoded] * min(bottleneck, X_encoded.ndim))
                X_enc_in = X_enc_in[:, :bottleneck] if X_enc_in.shape[1] >= bottleneck else X_enc_in

            dec = MLPRegressor(
                hidden_layer_sizes=(n_feat * 2,),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
            )
            dec.fit(X_enc_in, X_ae)
            X_reconstructed = dec.predict(X_enc_in)

            # ── Reconstruction Error per sampel ───────────────────────
            recon_error = np.mean((X_ae - X_reconstructed) ** 2, axis=1)
            threshold   = np.percentile(recon_error, threshold_pct)
            anomaly_idx = np.where(recon_error > threshold)[0]

            df_ae['reconstruction_error'] = recon_error
            df_ae['anomali'] = recon_error > threshold
            df_ae['nama_bulan'] = df_ae['bulan'].map(MAPPING_BULAN)

            # ── Plot reconstruction error ─────────────────────────────
            fig_ae = go.Figure()
            fig_ae.add_trace(go.Bar(
                x=df_ae['nama_bulan'],
                y=df_ae['reconstruction_error'],
                marker_color=[
                    'crimson' if a else 'steelblue'
                    for a in df_ae['anomali']
                ],
                name='Reconstruction Error',
            ))
            fig_ae.add_hline(
                y=threshold,
                line_dash='dash',
                line_color='orange',
                annotation_text=f"Threshold ({threshold_pct}th pct) = {threshold:.4f}",
            )
            fig_ae.update_layout(
                title="Autoencoder – Reconstruction Error per Bulan",
                xaxis_title="Bulan",
                yaxis_title="Reconstruction Error (MSE)",
                showlegend=False,
            )
            st.plotly_chart(fig_ae, use_container_width=True)

            # ── Tabel anomali ─────────────────────────────────────────
            st.subheader("🚨 Bulan Terdeteksi Anomali")
            df_anomali = df_ae[df_ae['anomali']][
                ['nama_bulan', 'jumlah', 'reconstruction_error']
            ].copy()
            df_anomali.columns = ['Bulan', 'Jumlah Permohonan', 'Reconstruction Error']
            df_anomali['Reconstruction Error'] = df_anomali['Reconstruction Error'].round(6)

            if df_anomali.empty:
                st.info("Tidak ada anomali terdeteksi pada threshold saat ini.")
            else:
                st.dataframe(df_anomali, use_container_width=True)
                for _, row in df_anomali.iterrows():
                    st.warning(
                        f"⚠️ **{row['Bulan']}** — {int(row['Jumlah Permohonan']):,} permohonan "
                        f"(error: {row['Reconstruction Error']:.5f})"
                    )

            # ── Representasi laten (encoded) ──────────────────────────
            st.subheader("🔵 Visualisasi Ruang Laten (Representasi Terkompresi)")
            if bottleneck >= 2:
                fig_lat = px.scatter(
                    x=X_encoded[:, 0] if X_encoded.ndim > 1 else X_encoded,
                    y=X_encoded[:, 1] if (X_encoded.ndim > 1 and X_encoded.shape[1] > 1)
                      else recon_error,
                    color=['Anomali' if a else 'Normal' for a in df_ae['anomali']],
                    text=df_ae['nama_bulan'],
                    color_discrete_map={'Anomali': 'crimson', 'Normal': 'steelblue'},
                    title="Ruang Laten Autoencoder (Dimensi 1 vs 2)",
                    labels={'x': 'Latent Dim 1', 'y': 'Latent Dim 2'},
                )
                fig_lat.update_traces(textposition='top center')
                st.plotly_chart(fig_lat, use_container_width=True)
            else:
                fig_lat1d = px.scatter(
                    x=df_ae['nama_bulan'],
                    y=X_encoded if X_encoded.ndim == 1 else X_encoded[:, 0],
                    color=['Anomali' if a else 'Normal' for a in df_ae['anomali']],
                    color_discrete_map={'Anomali': 'crimson', 'Normal': 'steelblue'},
                    title="Representasi Laten 1D per Bulan",
                    labels={'x': 'Bulan', 'y': 'Nilai Laten'},
                )
                st.plotly_chart(fig_lat1d, use_container_width=True)

# ───────────────────────────────────────────
# TAB 2 – JENIS TRANSAKSI
# ───────────────────────────────────────────
with tab2:
    st.header("Analisis Jenis Transaksi")

    if 'jenis_transaksi' not in df_f.columns:
        st.warning("Kolom 'jenis_transaksi' tidak ditemukan.")
    else:
        transaksi_counts = (
            df_f['jenis_transaksi']
            .value_counts()
            .reset_index()
        )
        transaksi_counts.columns = ['jenis_transaksi', 'jumlah']
        transaksi_counts['persentase (%)'] = (
            transaksi_counts['jumlah'] /
            transaksi_counts['jumlah'].sum() * 100
        ).round(2)

        st.dataframe(transaksi_counts, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_bar = px.bar(
                transaksi_counts, x='jenis_transaksi', y='jumlah',
                color='jenis_transaksi',
                title="Distribusi Jenis Transaksi"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            fig_pie = px.pie(
                transaksi_counts,
                names='jenis_transaksi',
                values='jumlah',
                title="Persentase Jenis Transaksi"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Tren per bulan
        st.subheader("Tren Jenis Transaksi per Bulan")
        transaksi_bulanan = (
            df_f.groupby(['bulan', 'jenis_transaksi'])
            .size()
            .unstack()
            .fillna(0)
            .reset_index()
        )
        transaksi_long = transaksi_bulanan.melt(
            id_vars='bulan', var_name='Jenis', value_name='Jumlah'
        )
        fig_tren = px.line(
            transaksi_long, x='bulan', y='Jumlah',
            color='Jenis', markers=True
        )
        fig_tren.update_xaxes(tickvals=list(range(1, 13)))
        st.plotly_chart(fig_tren, use_container_width=True)

        # Heatmap jenis transaksi × bulan
        st.subheader("Heatmap Permohonan per Bulan × Jenis Transaksi")
        df_clean = df_f[df_f['bulan'].between(1, 12)].copy()
        df_clean['bulan_nama'] = df_clean['bulan'].map(MAPPING_BULAN)
        heatmap_data = (
            df_clean.groupby(['bulan_nama', 'jenis_transaksi'])
            .size()
            .reset_index(name='jumlah')
        )
        urutan_bulan = list(MAPPING_BULAN.values())
        heatmap_data['bulan_nama'] = pd.Categorical(
            heatmap_data['bulan_nama'], categories=urutan_bulan, ordered=True
        )
        fig_heat = px.density_heatmap(
            heatmap_data, x='jenis_transaksi', y='bulan_nama', z='jumlah',
            color_continuous_scale='RdYlGn_r',
            title="Heatmap Permohonan Listrik"
        )
        fig_heat.update_traces(
            hovertemplate="<b>Bulan:</b> %{y}<br><b>Jenis:</b> %{x}<br><b>Jumlah:</b> %{z}<extra></extra>"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ───────────────────────────────────────────
# TAB 3 – ANALISIS SPASIAL
# ───────────────────────────────────────────
with tab3:
    st.header("Analisis Spasial – Kecamatan Fokus")

    if df_fok_f.empty:
        st.warning("Tidak ada data untuk kombinasi filter saat ini.")
    else:
        pivot = pd.crosstab(
            df_fok_f['kecamatan'],
            df_fok_f['jenis_transaksi']
        ).reset_index()

        if 'PASANG BARU' not in pivot.columns:
            pivot['PASANG BARU'] = 0
        if 'TAMBAH DAYA' not in pivot.columns:
            pivot['TAMBAH DAYA'] = 0

        jenis_cols = [c for c in pivot.columns if c != 'kecamatan']
        pivot['TOTAL'] = pivot[jenis_cols].sum(axis=1)
        pivot['RANK']  = pivot['TOTAL'].rank(ascending=False).astype(int)
        pivot = pivot.sort_values('TOTAL', ascending=False)

        st.subheader("Ranking Total Permohonan per Kecamatan")
        st.dataframe(pivot, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_hbar = px.bar(
                pivot.sort_values('TOTAL'),
                x='TOTAL', y='kecamatan',
                orientation='h',
                title="Total Permohonan",
                color='TOTAL', color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_hbar, use_container_width=True)

        with c2:
            fig_pb_td = px.bar(
                pivot, x='kecamatan',
                y=jenis_cols,
                barmode='group',
                title="Pasang Baru vs Tambah Daya per Kecamatan"
            )
            fig_pb_td.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_pb_td, use_container_width=True)

        # Tren bulanan per kecamatan
        st.subheader("Tren Bulanan per Kecamatan")
        tren_kec = (
            df_fok_f.groupby(['bulan', 'kecamatan'])
            .size()
            .reset_index(name='jumlah')
        )
        fig_tren_k = px.line(
            tren_kec, x='bulan', y='jumlah',
            color='kecamatan', markers=True
        )
        fig_tren_k.update_xaxes(tickvals=list(range(1, 13)))
        st.plotly_chart(fig_tren_k, use_container_width=True)

        # Rata-rata growth rate
        st.subheader("Rata-rata Growth Rate per Kecamatan (%)")
        tren_kec_pivot = (
            df_fok_f.groupby(['bulan', 'kecamatan'])
            .size()
            .unstack()
            .fillna(0)
        )
        growth_kec  = tren_kec_pivot.pct_change() * 100
        rata_growth = growth_kec.mean().sort_values(ascending=False).reset_index()
        rata_growth.columns = ['kecamatan', 'rata_growth_%']

        fig_gr = px.bar(
            rata_growth.sort_values('rata_growth_%'),
            x='rata_growth_%', y='kecamatan',
            orientation='h',
            color='rata_growth_%',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_gr, use_container_width=True)

        # Analisis selisih daya
        st.subheader("Distribusi Selisih Daya (VA)")
        fig_hist, ax_h = plt.subplots()
        ax_h.hist(df_fok_f['selisih_daya'].dropna(), bins=20, color='steelblue', edgecolor='white')
        ax_h.set_xlabel("Selisih Daya (VA)")
        ax_h.set_ylabel("Jumlah Pelanggan")
        ax_h.grid(True, alpha=0.4)
        st.pyplot(fig_hist)
        plt.close()

        rata_bulanan = (
            df_fok_f.groupby('bulan')['selisih_daya']
            .mean()
            .reset_index()
            .sort_values('bulan')
        )
        fig_rd = px.line(
            rata_bulanan, x='bulan', y='selisih_daya',
            markers=True,
            title="Rata-rata Kenaikan Daya per Bulan (VA)"
        )
        fig_rd.update_xaxes(tickvals=list(range(1, 13)))
        st.plotly_chart(fig_rd, use_container_width=True)

# ───────────────────────────────────────────
# TAB 4 – PETA INTERAKTIF
# ───────────────────────────────────────────
with tab4:
    st.header("🗺️ Peta Interaktif Kecamatan")

    if df_fok_f.empty:
        st.warning("Tidak ada data untuk filter saat ini.")
    else:
        pivot_map = pd.crosstab(
            df_fok_f['kecamatan'],
            df_fok_f['jenis_transaksi']
        ).reset_index()

        if 'PASANG BARU' not in pivot_map.columns:
            pivot_map['PASANG BARU'] = 0
        if 'TAMBAH DAYA' not in pivot_map.columns:
            pivot_map['TAMBAH DAYA'] = 0

        jenis_cols_map = [c for c in pivot_map.columns if c != 'kecamatan']
        pivot_map['TOTAL'] = pivot_map[jenis_cols_map].sum(axis=1)
        pivot_map['% PASANG BARU'] = (
            pivot_map['PASANG BARU'] / pivot_map['TOTAL'].replace(0, np.nan) * 100
        ).round(1).fillna(0)
        pivot_map['% TAMBAH DAYA'] = (
            pivot_map['TAMBAH DAYA'] / pivot_map['TOTAL'].replace(0, np.nan) * 100
        ).round(1).fillna(0)

        Q1 = pivot_map['TOTAL'].quantile(0.25)
        Q3 = pivot_map['TOTAL'].quantile(0.75)

        def kategori(t):
            if t >= Q3: return "High Demand"
            if t <= Q1: return "Low Demand"
            return "Medium Demand"

        pivot_map['Kategori'] = pivot_map['TOTAL'].apply(kategori)

        warna_map = {
            "High Demand":   "red",
            "Medium Demand": "orange",
            "Low Demand":    "green",
        }

        peta = folium.Map(location=[-7.26, 112.77], zoom_start=12, tiles='CartoDB positron')

        for _, row in pivot_map.iterrows():
            kec = row['kecamatan']
            if kec not in KOORDINAT:
                continue
            warna = warna_map[row['Kategori']]
            popup_html = f"""
            <b>{kec}</b><br><br>
            Pasang Baru : {row['PASANG BARU']} ({row['% PASANG BARU']}%)<br>
            Tambah Daya : {row['TAMBAH DAYA']} ({row['% TAMBAH DAYA']}%)<br>
            Total       : {row['TOTAL']}<br><br>
            <b>Kategori: {row['Kategori']}</b>
            """
            radius = max(row['TOTAL'] / 200, 5)
            folium.CircleMarker(
                location=KOORDINAT[kec],
                radius=radius,
                color=warna,
                fill=True,
                fill_color=warna,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(peta)

        legend_html = """
        <div style="position:fixed;top:20px;right:20px;width:160px;
        background:white;border:2px solid grey;z-index:9999;font-size:12px;padding:10px;">
        <b>Kategori Demand</b><br>
        <span style="color:red;">●</span> High Demand<br>
        <span style="color:orange;">●</span> Medium Demand<br>
        <span style="color:green;">●</span> Low Demand
        </div>
        """
        peta.get_root().html.add_child(folium.Element(legend_html))

        st_folium(peta, width=900, height=520)

# ───────────────────────────────────────────
# TAB 5 – CLUSTERING (K-Means)
# ───────────────────────────────────────────
with tab5:
    st.header("🤖 Spatial Clustering K-Means")

    if df_fok_f.empty or df_fok_f['kecamatan'].nunique() < 3:
        st.warning("Data tidak cukup untuk clustering (min 3 kecamatan).")
    else:
        agregasi = df_fok_f.groupby('kecamatan').agg(
            rata_selisi_daya=('selisih_daya', 'mean'),
            total_permohonan=('id_pelanggan', 'count')
        ).reset_index()

        all_jenis = df_fok_f['jenis_transaksi'].unique()
        nama_pb = next((j for j in all_jenis if 'PASANG' in j), None)
        nama_td = next((j for j in all_jenis if 'TAMBAH' in j or 'PERUBAHAN' in j or 'UBAH' in j), None)

        pb = df_fok_f[df_fok_f['jenis_transaksi'] == nama_pb].groupby('kecamatan').size() if nama_pb else pd.Series(dtype=int)
        td = df_fok_f[df_fok_f['jenis_transaksi'] == nama_td].groupby('kecamatan').size() if nama_td else pd.Series(dtype=int)

        agregasi['pasang_baru'] = agregasi['kecamatan'].map(pb).fillna(0)
        agregasi['tambah_daya'] = agregasi['kecamatan'].map(td).fillna(0)

        features = ['total_permohonan', 'pasang_baru', 'tambah_daya', 'rata_selisi_daya']
        cluster_data = agregasi[features].fillna(0)

        if len(cluster_data) >= 3:
            scaler      = StandardScaler()
            scaled      = scaler.fit_transform(cluster_data)
            n_clusters  = min(3, len(cluster_data))
            kmeans      = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            agregasi['CLUSTER'] = kmeans.fit_predict(scaled)

            rank = (
                agregasi.groupby('CLUSTER')['total_permohonan']
                .mean()
                .sort_values()
            )
            label_map = {
                rank.index[0]: 'Low Demand',
                rank.index[-1]: 'High Demand',
            }
            if n_clusters == 3:
                label_map[rank.index[1]] = 'Medium Demand'

            agregasi['KATEGORI'] = agregasi['CLUSTER'].map(label_map)

            st.subheader("Hasil Clustering per Kecamatan")
            st.dataframe(
                agregasi[['kecamatan', 'total_permohonan', 'pasang_baru', 'tambah_daya', 'KATEGORI']],
                use_container_width=True
            )

            fig_cl = px.bar(
                agregasi.sort_values('total_permohonan'),
                x='total_permohonan', y='kecamatan',
                color='KATEGORI',
                orientation='h',
                color_discrete_map={
                    'High Demand': 'red',
                    'Medium Demand': 'orange',
                    'Low Demand': 'green',
                },
                title="Cluster Demand Kecamatan"
            )
            st.plotly_chart(fig_cl, use_container_width=True)

            # Peta cluster
            st.subheader("Peta Cluster Kecamatan")
            peta_cl = folium.Map(location=[-7.26, 112.77], zoom_start=12, tiles='CartoDB positron')
            warna_cl = {'High Demand': 'red', 'Medium Demand': 'orange', 'Low Demand': 'green'}

            for _, row in agregasi.iterrows():
                kec = row['kecamatan']
                if kec not in KOORDINAT:
                    continue
                warna = warna_cl.get(row['KATEGORI'], 'gray')
                popup_html = f"""
                <b>{kec}</b><br><br>
                <b>Kategori:</b> {row['KATEGORI']}<br>
                Total       : {row['total_permohonan']}<br>
                Pasang Baru : {row['pasang_baru']}<br>
                Tambah Daya : {row['tambah_daya']}<br>
                Selisih Daya: {round(row['rata_selisi_daya'], 2)} VA
                """
                folium.CircleMarker(
                    location=KOORDINAT[kec],
                    radius=max(row['total_permohonan'] / 200, 5),
                    color=warna, fill=True, fill_color=warna, fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=250),
                ).add_to(peta_cl)

            legend_html2 = """
            <div style="position:fixed;top:20px;right:20px;width:160px;
            background:white;border:2px solid grey;z-index:9999;font-size:12px;padding:10px;">
            <b>Cluster Demand</b><br>
            <span style="color:red;">●</span> High Demand<br>
            <span style="color:orange;">●</span> Medium Demand<br>
            <span style="color:green;">●</span> Low Demand
            </div>
            """
            peta_cl.get_root().html.add_child(folium.Element(legend_html2))
            st_folium(peta_cl, width=900, height=500)
        else:
            st.info("Jumlah kecamatan tidak cukup untuk clustering.")

# ───────────────────────────────────────────
# TAB 6 – GANTT CHART (Perencanaan Program)
# ───────────────────────────────────────────
with tab6:
    st.header("📋 Perencanaan Program – Gantt Chart")
    st.markdown(
        "Gantt chart di bawah menggambarkan jadwal kegiatan program magang/analisis "
        "permohonan listrik dari **Januari hingga Mei 2025**."
    )

    # ── Data tugas (bisa diedit di sini) ────────────────────────────────
    tasks = [
        # (Tugas, Fase, Mulai, Selesai)
        ("Pengumpulan Data",        "Persiapan",   "2025-01-06", "2025-01-17"),
        ("Pembersihan Data",        "Persiapan",   "2025-01-13", "2025-01-24"),
        ("Eksplorasi Data (EDA)",   "Analisis",    "2025-01-20", "2025-02-07"),
        ("Analisis Temporal",       "Analisis",    "2025-02-03", "2025-02-21"),
        ("Analisis Spasial",        "Analisis",    "2025-02-17", "2025-03-07"),
        ("Clustering K-Means",      "Pemodelan",   "2025-03-03", "2025-03-21"),
        ("Optimasi Klaster",        "Pemodelan",   "2025-03-17", "2025-04-04"),
        ("Graph Network Analysis",  "Pemodelan",   "2025-04-01", "2025-04-18"),
        ("Pembuatan Dashboard",     "Visualisasi", "2025-04-07", "2025-04-25"),
        ("Uji Coba & QA",           "Visualisasi", "2025-04-21", "2025-05-09"),
        ("Laporan Akhir",           "Pelaporan",   "2025-05-05", "2025-05-23"),
        ("Presentasi Hasil",        "Pelaporan",   "2025-05-19", "2025-05-30"),
    ]

    df_gantt = pd.DataFrame(tasks, columns=["Task", "Phase", "Start", "Finish"])
    df_gantt["Start"]  = pd.to_datetime(df_gantt["Start"])
    df_gantt["Finish"] = pd.to_datetime(df_gantt["Finish"])

    color_map = {
        "Persiapan":   "#4C78A8",
        "Analisis":    "#F58518",
        "Pemodelan":   "#54A24B",
        "Visualisasi": "#E45756",
        "Pelaporan":   "#B279A2",
    }

    fig_gantt = px.timeline(
        df_gantt,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Phase",
        color_discrete_map=color_map,
        title="Jadwal Program Analisis Permohonan Listrik (Jan – Mei 2025)",
        hover_data={"Start": "|%d %b %Y", "Finish": "|%d %b %Y", "Phase": True},
    )
    fig_gantt.update_yaxes(autorange="reversed")
    fig_gantt.update_xaxes(
        range=["2025-01-01", "2025-05-31"],
        dtick="M1",
        tickformat="%b %Y",
    )
    fig_gantt.update_layout(
        height=520,
        legend_title_text="Fase",
        xaxis_title="",
        yaxis_title="",
    )
    st.plotly_chart(fig_gantt, use_container_width=True)

    # Tabel ringkasan
    st.subheader("📄 Ringkasan Jadwal")
    df_show = df_gantt.copy()
    df_show["Start"]  = df_show["Start"].dt.strftime("%d %b %Y")
    df_show["Finish"] = df_show["Finish"].dt.strftime("%d %b %Y")
    df_show["Durasi (hari)"] = (
        pd.to_datetime(df_gantt["Finish"]) - pd.to_datetime(df_gantt["Start"])
    ).dt.days
    st.dataframe(df_show[["Phase", "Task", "Start", "Finish", "Durasi (hari)"]],
                 use_container_width=True)

    # Progress per fase
    st.subheader("📊 Durasi Total per Fase")
    fase_durasi = df_gantt.copy()
    fase_durasi["durasi"] = (fase_durasi["Finish"] - fase_durasi["Start"]).dt.days
    fase_sum = fase_durasi.groupby("Phase")["durasi"].sum().reset_index()
    fase_sum.columns = ["Fase", "Total Hari"]
    fig_fase = px.bar(
        fase_sum, x="Fase", y="Total Hari",
        color="Fase", color_discrete_map=color_map,
        text="Total Hari",
        title="Total Durasi per Fase (hari)"
    )
    fig_fase.update_traces(textposition="outside")
    st.plotly_chart(fig_fase, use_container_width=True)

# ───────────────────────────────────────────
# TAB 7 – TEKNIK OPTIMASI (Elbow + Silhouette)
# ───────────────────────────────────────────
with tab7:
    st.header("📐 Teknik Optimasi – Elbow Method & Silhouette Score")
    st.markdown(
        "Analisis ini membantu menentukan **jumlah klaster optimal (k)** untuk "
        "K-Means menggunakan dua metode: **Elbow** dan **Silhouette**."
    )

    if df_fok_f.empty or df_fok_f['kecamatan'].nunique() < 3:
        st.warning("Data kecamatan tidak cukup (min 3 kecamatan). Sesuaikan filter.")
    else:
        # Siapkan data agregasi per kecamatan (sama dengan tab clustering)
        agg_opt = df_fok_f.groupby('kecamatan').agg(
            rata_selisi_daya=('selisih_daya', 'mean'),
            total_permohonan=('id_pelanggan', 'count')
        ).reset_index()

        all_j = df_fok_f['jenis_transaksi'].unique()
        n_pb  = next((j for j in all_j if 'PASANG' in j), None)
        n_td  = next((j for j in all_j if 'TAMBAH' in j or 'UBAH' in j), None)
        pb_s  = df_fok_f[df_fok_f['jenis_transaksi'] == n_pb].groupby('kecamatan').size() if n_pb else pd.Series(dtype=int)
        td_s  = df_fok_f[df_fok_f['jenis_transaksi'] == n_td].groupby('kecamatan').size() if n_td else pd.Series(dtype=int)
        agg_opt['pasang_baru'] = agg_opt['kecamatan'].map(pb_s).fillna(0)
        agg_opt['tambah_daya'] = agg_opt['kecamatan'].map(td_s).fillna(0)

        feat_opt = ['total_permohonan', 'pasang_baru', 'tambah_daya', 'rata_selisi_daya']
        X_opt    = agg_opt[feat_opt].fillna(0)
        sc_opt   = StandardScaler()
        X_scaled = sc_opt.fit_transform(X_opt)

        max_k = min(len(X_opt) - 1, 8)

        if max_k < 2:
            st.info("Data terlalu sedikit untuk menghitung optimasi (butuh ≥ 3 titik).")
        else:
            k_range    = range(2, max_k + 1)
            inertias   = []
            sil_scores = []

            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
                inertias.append(km.inertia_)
                sil_scores.append(silhouette_score(X_scaled, labels))

            c1, c2 = st.columns(2)

            # ── Elbow ────────────────────────────────────────────────
            with c1:
                st.subheader("📉 Elbow Method (Inertia)")
                fig_elbow, ax_el = plt.subplots(figsize=(6, 4))
                ax_el.plot(list(k_range), inertias, marker='o', color='steelblue', linewidth=2)
                ax_el.set_xlabel("Jumlah Klaster (k)")
                ax_el.set_ylabel("Inertia (Within-cluster SSE)")
                ax_el.set_title("Elbow Method")
                ax_el.grid(True, alpha=0.4)

                # Tandai titik elbow secara otomatis (max penurunan)
                deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
                best_k_elbow = list(k_range)[deltas.index(max(deltas)) + 1]
                ax_el.axvline(best_k_elbow, color='red', linestyle='--',
                              label=f"Elbow k={best_k_elbow}")
                ax_el.legend()
                st.pyplot(fig_elbow)
                plt.close()
                st.info(f"💡 Titik Elbow terdeteksi di **k = {best_k_elbow}**")

            # ── Silhouette ───────────────────────────────────────────
            with c2:
                st.subheader("📈 Silhouette Score")
                fig_sil, ax_si = plt.subplots(figsize=(6, 4))
                ax_si.plot(list(k_range), sil_scores, marker='s', color='darkorange', linewidth=2)
                ax_si.set_xlabel("Jumlah Klaster (k)")
                ax_si.set_ylabel("Silhouette Score")
                ax_si.set_title("Silhouette Analysis")
                ax_si.grid(True, alpha=0.4)

                best_k_sil = list(k_range)[sil_scores.index(max(sil_scores))]
                ax_si.axvline(best_k_sil, color='green', linestyle='--',
                              label=f"Best k={best_k_sil}")
                ax_si.legend()
                st.pyplot(fig_sil)
                plt.close()
                st.success(f"✅ Silhouette tertinggi di **k = {best_k_sil}** "
                           f"(score = {max(sil_scores):.4f})")

            # ── Tabel perbandingan ───────────────────────────────────
            st.subheader("📋 Tabel Perbandingan Inertia & Silhouette")
            df_opt_tbl = pd.DataFrame({
                "k (Klaster)":      list(k_range),
                "Inertia":          [round(v, 4) for v in inertias],
                "Silhouette Score": [round(v, 4) for v in sil_scores],
            })
            st.dataframe(df_opt_tbl, use_container_width=True)

            # ── Rekomendasi ──────────────────────────────────────────
            st.subheader("🎯 Rekomendasi Klaster Optimal")
            if best_k_elbow == best_k_sil:
                rek_k = best_k_elbow
                st.success(f"Kedua metode sepakat: **k = {rek_k}** adalah jumlah klaster optimal.")
            else:
                st.warning(
                    f"Elbow menyarankan **k = {best_k_elbow}**, "
                    f"Silhouette menyarankan **k = {best_k_sil}**. "
                    f"Disarankan menggunakan **k = {best_k_sil}** (Silhouette lebih intuitif)."
                )
                rek_k = best_k_sil

            # Jalankan K-Means dengan k rekomendasi dan tampilkan hasilnya
            km_rek = KMeans(n_clusters=rek_k, random_state=42, n_init=10)
            agg_opt['CLUSTER_OPT'] = km_rek.fit_predict(X_scaled)

            fig_opt_bar = px.bar(
                agg_opt.sort_values('total_permohonan'),
                x='total_permohonan', y='kecamatan',
                color='CLUSTER_OPT',
                orientation='h',
                title=f"Hasil K-Means dengan k={rek_k} (Optimal)",
                color_continuous_scale='Viridis',
            )
            st.plotly_chart(fig_opt_bar, use_container_width=True)

# ───────────────────────────────────────────
# TAB 8 – GRAPH SAINS DATA (NetworkX)
# ───────────────────────────────────────────
with tab8:
    st.header("🕸️ Graph Sains Data – Analisis Jaringan Kecamatan (NetworkX)")
    st.markdown(
        "Graf ini menggambarkan **kemiripan pola permintaan** antar kecamatan berdasarkan "
        "data permohonan listrik. Dua kecamatan terhubung jika korelasi Pearson "
        "total permohonan bulanan mereka **≥ threshold**."
    )

    if df_fok_f.empty or df_fok_f['kecamatan'].nunique() < 2:
        st.warning("Data tidak cukup untuk analisis jaringan.")
    else:
        # ── Slider threshold ─────────────────────────────────────────
        threshold = st.slider(
            "Threshold Korelasi (edge dibuat jika r ≥ threshold)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )

        # ── Pivot: baris = bulan, kolom = kecamatan ──────────────────
        pivot_net = (
            df_fok_f.groupby(['bulan', 'kecamatan'])
            .size()
            .unstack(fill_value=0)
        )

        # Hitung matriks korelasi antar kecamatan
        corr_matrix = pivot_net.corr()

        # ── Bangun graf ───────────────────────────────────────────────
        G = nx.Graph()
        kecamatan_nodes = list(corr_matrix.columns)
        G.add_nodes_from(kecamatan_nodes)

        edges_data = []
        for i, kec_a in enumerate(kecamatan_nodes):
            for j, kec_b in enumerate(kecamatan_nodes):
                if j <= i:
                    continue
                r = corr_matrix.loc[kec_a, kec_b]
                if r >= threshold:
                    G.add_edge(kec_a, kec_b, weight=round(float(r), 3))
                    edges_data.append((kec_a, kec_b, round(float(r), 3)))

        # ── Metrik jaringan ───────────────────────────────────────────
        st.subheader("📊 Metrik Jaringan")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jumlah Node",  G.number_of_nodes())
        m2.metric("Jumlah Edge",  G.number_of_edges())
        m3.metric("Densitas Graf", f"{nx.density(G):.3f}")
        avg_deg = (
            sum(d for _, d in G.degree()) / G.number_of_nodes()
            if G.number_of_nodes() > 0 else 0
        )
        m4.metric("Rata-rata Degree", f"{avg_deg:.2f}")

        # ── Visualisasi NetworkX dengan Matplotlib ────────────────────
        st.subheader("🕸️ Visualisasi Graf Kecamatan")

        fig_net, ax_net = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42, k=1.5)

        # Warna node berdasarkan degree
        degrees   = dict(G.degree())
        node_color = [degrees.get(n, 0) for n in G.nodes()]
        max_deg    = max(node_color) if node_color else 1

        # Tebal edge berdasarkan bobot
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_widths  = [w * 4 for w in edge_weights]

        nx.draw_networkx_nodes(
            G, pos, ax=ax_net,
            node_color=node_color,
            node_size=900,
            cmap=plt.cm.YlOrRd,
            vmin=0, vmax=max_deg,
        )
        nx.draw_networkx_labels(G, pos, ax=ax_net, font_size=8, font_weight='bold')
        nx.draw_networkx_edges(
            G, pos, ax=ax_net,
            width=edge_widths,
            edge_color=edge_weights,
            edge_cmap=plt.cm.Blues,
            alpha=0.7,
        )
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax_net, font_size=7)

        ax_net.set_title("Graf Kemiripan Permohonan Antar Kecamatan", fontsize=12)
        ax_net.axis('off')
        plt.tight_layout()
        st.pyplot(fig_net)
        plt.close()

        # ── Tabel edge ────────────────────────────────────────────────
        if edges_data:
            st.subheader("📋 Tabel Koneksi (Edge)")
            df_edges = pd.DataFrame(
                edges_data, columns=["Kecamatan A", "Kecamatan B", "Korelasi (r)"]
            ).sort_values("Korelasi (r)", ascending=False)
            st.dataframe(df_edges, use_container_width=True)
        else:
            st.info("Tidak ada edge yang terbentuk pada threshold ini. Coba turunkan threshold.")

        # ── Degree centrality ─────────────────────────────────────────
        st.subheader("🏆 Degree Centrality per Kecamatan")
        deg_cent = nx.degree_centrality(G)
        df_cent  = pd.DataFrame(
            deg_cent.items(), columns=["Kecamatan", "Degree Centrality"]
        ).sort_values("Degree Centrality", ascending=False)
        fig_cent = px.bar(
            df_cent, x="Kecamatan", y="Degree Centrality",
            color="Degree Centrality", color_continuous_scale="Viridis",
            title="Tingkat Keterhubungan (Degree Centrality) Kecamatan"
        )
        st.plotly_chart(fig_cent, use_container_width=True)

        # ── Heatmap korelasi ──────────────────────────────────────────
        st.subheader("🔥 Heatmap Korelasi Antar Kecamatan")
        fig_corr = px.imshow(
            corr_matrix.round(2),
            text_auto=True,
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Matriks Korelasi Pearson – Total Permohonan Bulanan",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.caption("Dashboard PLN – Analisis Permohonan Listrik · Dibuat dengan Streamlit")
