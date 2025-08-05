# pages/1_Eksplorasi_Data.py

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Eksplorasi Data", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    """Memuat dan membersihkan dataset."""
    data = pd.read_csv('formatted_output_2.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').fillna(0)
    data = data.sort_values('Date').reset_index(drop=True)
    data_cleaned = data.dropna()
    return data_cleaned

data = load_data()

st.title("📊 Eksplorasi Data Harga CPO")
st.write("Di halaman ini, Anda dapat melihat data mentah dan statistik deskriptifnya.")

# Tampilkan data mentah
st.header("Data Mentah Harga CPO", divider="rainbow")
st.dataframe(data)

# Tampilkan statistik deskriptif
st.header("Statistik Deskriptif", divider="rainbow")
st.dataframe(data.describe())