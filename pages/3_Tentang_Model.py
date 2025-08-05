# pages/3_Tentang_Model.py

import streamlit as st

st.set_page_config(page_title="Tentang Model", page_icon="🧠", layout="wide")

st.title("🧠 Tentang Model yang Digunakan")

st.header("Metodologi", divider="rainbow")
st.markdown("""
Model ini dibangun menggunakan **Support Vector Regression (SVR)**, sebuah teknik *machine learning* yang kuat untuk masalah regresi. SVR bekerja dengan menemukan "jalan" (hyperplane) terbaik yang memisahkan data dengan margin kesalahan sekecil mungkin.

Untuk mendapatkan performa terbaik, *hyperparameter* dari model SVR (seperti `C`, `gamma`, dan `epsilon`) dioptimalkan menggunakan algoritma **Particle Swarm Optimization (PSO)**. PSO adalah teknik optimasi yang terinspirasi dari perilaku kawanan burung atau ikan dalam mencari makanan.
""")

st.header("Performa Model", divider="rainbow")
st.info("Metrik berikut dihitung pada set data pengujian (data yang tidak pernah dilihat model saat training).")

# Anda bisa mengisi nilai ini secara manual dari hasil eksekusi notebook Anda
metrik = {
    "RMSE (Root Mean Squared Error)": 500.12, # Ganti dengan hasil Anda
    "MAE (Mean Absolute Error)": 292.46,      # Ganti dengan hasil Anda
    "R² Score": -0.5031                        # Ganti dengan hasil Anda
}
st.json(metrik)
st.warning("""
**Catatan tentang R² Score Negatif:** Nilai R² yang negatif menunjukkan bahwa model, dalam konfigurasinya saat ini, berkinerja lebih buruk daripada hanya menebak nilai rata-rata. Ini menyoroti pentingnya optimasi hyperparameter dan rekayasa fitur lebih lanjut untuk meningkatkan akurasi.
""")

st.header("Hyperparameter Terbaik Hasil PSO", divider="rainbow")
# Ganti dengan parameter terbaik dari hasil optimasi Anda
best_params = {
    "C": 2700,
    "gamma": 0.01,
    "epsilon": 0.004
}
st.json(best_params)