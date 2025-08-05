# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import timedelta

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Peramalan Harga CPO",
    page_icon="🌴",
    layout="wide"
)

# --- FUNGSI-FUNGSI BANTU ---
# Gunakan cache agar pemuatan data dan model lebih cepat
@st.cache_data
def load_data():
    """Memuat dan membersihkan dataset."""
    data = pd.read_csv('formatted_output_2.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0)
    data = data.sort_values('Date').reset_index(drop=True)
    data_cleaned = data.dropna()
    return data_cleaned

@st.cache_data
def load_model_and_objects():
    """Memuat model dan objek lain yang tersimpan."""
    model = joblib.load('model_svr.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_Y = joblib.load('scaler_Y.pkl')
    return model, scaler_X, scaler_Y

# --- PEMUATAN DATA & MODEL ---
data = load_data()
model, scaler_X, scaler_Y = load_model_and_objects()

# --- TAMPILAN DASHBOARD ---
st.title("🌴 Dashboard Peramalan Harga CPO")
st.write(
    "Selamat datang di dashboard interaktif untuk peramalan harga Crude Palm Oil (CPO). "
    "Dashboard ini menggunakan model Support Vector Regression (SVR) yang dioptimalkan dengan Particle Swarm Optimization (PSO) "
    "untuk memberikan wawasan tentang data historis dan potensi harga di masa depan."
)

# --- METRIK UTAMA ---
st.header("Ringkasan Harga Terkini", divider='rainbow')

# Ambil data terakhir
last_data = data.iloc[-1]
prev_last_data = data.iloc[-2]

# Prediksi untuk hari berikutnya
# Siapkan fitur untuk prediksi besok (mirip logika di horizon)
# Kita butuh DataFrame fitur lengkap untuk prediksi tunggal ini
# (Untuk kesederhanaan, kita tampilkan saja data terakhir)

col1, col2, col3 = st.columns(3)
col1.metric(
    label="Harga Penutupan Terakhir",
    value=f"RM {last_data['Close']:.2f}",
    delta=f"{last_data['Close'] - prev_last_data['Close']:.2f} (vs kemarin)"
)
col2.metric(
    label="Volume Perdagangan Terakhir",
    value=f"{int(last_data['Volume']):,}",
    delta=f"{int(last_data['Volume'] - prev_last_data['Volume']):,}"
)
col3.metric(
    label="Tanggal Data Terakhir",
    value=last_data['Date'].strftime('%d %B %Y')
)

# --- GRAFIK UTAMA ---
st.header("Grafik Harga Historis CPO", divider='rainbow')
fig = go.Figure()

# Tambahkan trace untuk harga historis
fig.add_trace(go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name='Harga Historis'
))

# Konfigurasi layout
fig.update_layout(
    title='Pergerakan Harga CPO dari Waktu ke Waktu',
    yaxis_title='Harga (MYR per tonne)',
    xaxis_title='Tanggal',
    xaxis_rangeslider_visible=True # Menambahkan slider rentang tanggal
)

st.plotly_chart(fig, use_container_width=True)


st.sidebar.success("Pilih halaman di atas untuk navigasi.")