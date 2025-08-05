# pages/2_Peramalan_Interaktif.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Peramalan Interaktif", page_icon="🔮", layout="wide")

# --- FUNGSI-FUNGSI BANTU (TETAP SAMA) ---
@st.cache_data
def load_data():
    """Memuat dan membersihkan dataset."""
    data = pd.read_csv('formatted_output_2.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0)
    data = data.sort_values('Date').reset_index(drop=True)
    return data.dropna()

@st.cache_data
def load_model_and_objects():
    """Memuat model, scaler, dan juga daftar nama fitur."""
    model = joblib.load('model_svr.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_Y = joblib.load('scaler_Y.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler_X, scaler_Y, feature_names

# --- FUNGSI INTI PERAMALAN (TETAP SAMA) ---
def predict_future_horizon(model, scalerX, scalerY, feature_names, initial_features_unscaled, horizon, last_known_date):
    """Memprediksi harga untuk horizon waktu ke depan menggunakan metode iteratif."""
    initial_features_df = pd.DataFrame([initial_features_unscaled])
    current_features_scaled = scalerX.transform(initial_features_df)
    
    future_predictions_scaled = []
    last_volume = initial_features_unscaled.get('Prev_Volume', 0)

    for i in range(horizon):
        next_pred_scaled = model.predict(current_features_scaled)[0]
        future_predictions_scaled.append(next_pred_scaled)
        
        next_pred_unscaled = scalerY.inverse_transform(np.array([[next_pred_scaled]]))[0][0]
        
        next_date = last_known_date + pd.Timedelta(days=i + 1)
        
        next_features_unscaled = pd.Series(index=feature_names)
        next_features_unscaled['Year'] = next_date.year
        next_features_unscaled['Month'] = next_date.month
        next_features_unscaled['Day_of_Month'] = next_date.day
        next_features_unscaled['Day_of_Year'] = next_date.dayofyear
        next_features_unscaled['Week_of_Year'] = next_date.isocalendar().week
        next_features_unscaled['Day_of_Week'] = next_date.isoweekday()
        next_features_unscaled['Weekday'] = next_date.weekday()
        next_features_unscaled['Quarter'] = next_date.quarter
        
        next_features_unscaled['Open'] = next_pred_unscaled
        next_features_unscaled['Prev_High'] = next_pred_unscaled
        next_features_unscaled['Prev_Low'] = next_pred_unscaled
        if 'Prev_Volume' in feature_names:
            next_features_unscaled['Prev_Volume'] = last_volume
        if 'Prev_Percentage Change' in feature_names:
            next_features_unscaled['Prev_Percentage Change'] = 0.0

        next_features_unscaled.fillna(0, inplace=True)

        next_features_df = pd.DataFrame([next_features_unscaled])
        current_features_scaled = scalerX.transform(next_features_df)

    future_predictions_unscaled = scalerY.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({'Date': future_dates, 'Prediction': future_predictions_unscaled.flatten()}).set_index('Date')


# --- PEMUATAN DATA & MODEL ---
data = load_data()
model, scaler_X, scaler_Y, feature_names = load_model_and_objects()


# --- TAMPILAN DASHBOARD ---
st.title("🔮 Peramalan Harga Interaktif")
st.write("Gunakan slider di bawah untuk memilih berapa hari ke depan Anda ingin melakukan peramalan harga CPO.")

horizon = st.slider("Pilih Horizon Prediksi (hari):", min_value=7, max_value=365, value=30, step=1)

if st.button("Jalankan Peramalan"):
    with st.spinner("Mempersiapkan data dan melakukan peramalan..."):
        # --- PERBAIKAN UTAMA DI SINI ---
        # Logika ini sekarang mereplikasi proses dari notebook secara akurat.
        
        # 1. Buat data lag dari data historis
        df_lag = data.copy()
        # Kolom yang akan di-lag adalah semua KECUALI yang berhubungan dengan target hari ini ('Open', 'Date', 'Close')
        cols_to_lag = [col for col in df_lag.columns if col not in ['Open', 'Date', 'Close']]
        
        for col in cols_to_lag:
            df_lag[f'Prev_{col}'] = df_lag[col].shift(periods=1)

        # 2. Hapus baris pertama (yang berisi NaN) dan kolom asli yang sudah tidak perlu
        df_lag = df_lag.iloc[1:, :]
        df_lag = df_lag.drop(columns=cols_to_lag)

        # 3. Buat fitur tanggal
        newdate = df_lag['Date']
        df_date = pd.DataFrame({
            'Year': newdate.dt.year,
            'Month': newdate.dt.month,
            'Day_of_Month': newdate.dt.day,
            'Day_of_Year': newdate.dt.dayofyear,
            'Week_of_Year': newdate.dt.isocalendar().week,
            'Day_of_Week': newdate.dt.weekday + 1,
            'Weekday': newdate.dt.weekday,
            'Quarter': newdate.dt.quarter,
        })

        # 4. Gabungkan semuanya menjadi DataFrame fitur final
        # Urutan kolom diatur agar sama dengan feature_names yang disimpan
        features_df = pd.concat([df_date.reset_index(drop=True), df_lag.drop(columns=['Date', 'Close']).reset_index(drop=True)], axis=1)
        features_df = features_df[feature_names] #<- Baris Kunci! Memastikan urutan kolom benar.

        # Ambil baris terakhir sebagai titik awal
        last_features_unscaled = features_df.iloc[-1]
        last_known_date = data['Date'].iloc[-1]
        
        # Lakukan peramalan
        df_future = predict_future_horizon(model, scaler_X, scaler_Y, feature_names, last_features_unscaled, horizon, last_known_date)

        # Tampilkan grafik
        st.header(f"Hasil Peramalan untuk {horizon} Hari ke Depan", divider="rainbow")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Harga Historis'))
        fig.add_trace(go.Scatter(x=df_future.index, y=df_future['Prediction'], mode='lines', name='Prediksi Masa Depan', line=dict(dash='dash')))
        fig.update_layout(title="Grafik Peramalan Harga CPO", yaxis_title="Harga (MYR)")
        st.plotly_chart(fig, use_container_width=True)

        # Tampilkan tabel prediksi
        st.header("Tabel Hasil Peramalan", divider="rainbow")
        st.dataframe(df_future)