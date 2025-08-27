import streamlit as st
from streamlit_folium import st_folium
import folium
from datetime import date
import joblib
from xgboost import XGBRegressor
import numpy as np
import json



st.set_page_config(layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #110827;
        color: #FFCFFF;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 16px;
    }
    .stSelectbox>div>div {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("# 🗺️ Geographic Product Demand Forecasting")
st.markdown("### 📍 Click a location on the map to predict product demand")


def process_data(lat, lon, option, date):
    cluster = joblib.load('../models/kmeans_model.pkl')
    xbg = joblib.load('../models/xbg_model.pkl')
    features = np.array([lat, lon]).reshape(1, -1)
    prid_cluster = cluster.predict(features)

    
    day = date.day
    month = date.month
    year = date.year
    day_of_week = date.weekday()
    is_weekend = int(day_of_week >= 5)

    st.success(f"🧭 Predicted Cluster: {prid_cluster[0]}")

    with open('../models/label_encoding.json', 'r') as f:
        loaded_dict = json.load(f)
    category = loaded_dict[option]

    features = [category, prid_cluster[0], lat, lon, year, month, day, day_of_week, is_weekend]
    features_array = np.array(features).reshape(1, -1)
    pred = xbg.predict(features_array)

    st.metric(label="💰 Predicted Sales Revenue", value=f"₹ {pred[0]:,.1f}")



with st.container():
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
    m.add_child(folium.LatLngPopup())
    st_data = st_folium(m, width=700, height=500)

# ----------------------------
# ⬇️ CONDITIONAL UI BELOW MAP
# ----------------------------
if st_data and st_data['last_clicked']:
    lat = st_data['last_clicked']['lat']
    lon = st_data['last_clicked']['lng']
    st.success(f"✅ Clicked location: Latitude: {lat:.6f}, Longitude: {lon:.6f}")

    
    options = [
        'Bed', 'Bluetooth Speaker', 'Board Game', 'Camera', 'Cereal', 'Coffee', 'Cookbook',
        'Dining Table', 'Doll', 'Energy Drinks', 'Formal Shirt', 'Gaming Console', 'Headphones',
        'Jeans', 'Laptop', 'Men T-shirt', 'Microwave', 'Novel', 'Organic Snacks', 'Protein Bars',
        'Puzzle', 'Remote Control Car', 'Running Shoes', 'Self-help Book', 'Smartphone',
        'Smartwatch', 'Sofa', 'Sunglasses', 'Sweaters', 'Tablet', 'Textbook', 'Toy Car',
        'Washing Machine', 'Winter Jacket', 'Women Dress'
    ]

    
    icon_map = {
        'Bed': '🛏️', 'Bluetooth Speaker': '🔊', 'Board Game': '🎲', 'Camera': '📷',
        'Cereal': '🥣', 'Coffee': '☕', 'Cookbook': '📚', 'Dining Table': '🍽️',
        'Doll': '🪆', 'Energy Drinks': '⚡', 'Formal Shirt': '👔', 'Gaming Console': '🎮',
        'Headphones': '🎧', 'Jeans': '👖', 'Laptop': '💻', 'Men T-shirt': '👕',
        'Microwave': '🔥', 'Novel': '📖', 'Organic Snacks': '🍏', 'Protein Bars': '💪',
        'Puzzle': '🧩', 'Remote Control Car': '🚗', 'Running Shoes': '👟',
        'Self-help Book': '📘', 'Smartphone': '📱', 'Smartwatch': '⌚',
        'Sofa': '🛋️', 'Sunglasses': '🕶️', 'Sweaters': '🧥', 'Tablet': '📱',
        'Textbook': '📚', 'Toy Car': '🚙', 'Washing Machine': '🧺',
        'Winter Jacket': '🧥', 'Women Dress': '👗'
    }

    # Layout using columns
    col1, col2 = st.columns(2)

    with col1:
        show_icons = st.checkbox("Show icons in product list", value=True)

    if show_icons:
        display_options = [f"{icon_map.get(opt, '')} {opt}" for opt in options]
    else:
        display_options = options

    selected_display_option = st.selectbox("Select a product option:", display_options)
    selected_option = selected_display_option.split(' ', 1)[-1]

    with col2:
        selected_date = st.date_input("Select a date:", date.today())

    
    if st.button("🔍 Submit for Forecast"):
        process_data(lat, lon, selected_option, selected_date)


# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center><sub>🔍  Model Used : XGBoost | Built By Tejas Pawar </sub></center>", unsafe_allow_html=True)
