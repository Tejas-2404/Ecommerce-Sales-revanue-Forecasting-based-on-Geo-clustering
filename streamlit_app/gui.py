import streamlit as st
from streamlit_folium import st_folium
import folium
from datetime import date
import joblib
from xgboost import XGBRegressor
import numpy as np
import json

st.title("Geographic Product Demand Forecasting")

def process_data(lat, lon, option, date):
    cluster = joblib.load('../models/kmeans_model.pkl')
    xbg=joblib.load('../models/xbg_model.pkl')
    features = np.array([lat, lon])         
    features = features.reshape(1, -1)
    prid_cluster=cluster.predict(features)
    day = date.day                
    month = date.month         
    year = date.year             
    day_of_week = date.weekday()  
    is_weekend = int(day_of_week >= 5)
    st.write(f"Cluster: {prid_cluster[0]}")
    with open('../models/label_encoding.json', 'r') as f:
        loaded_dict = json.load(f)
    catagory=loaded_dict[option]
    features=[ catagory, prid_cluster[0], lat, lon, year, month, day, day_of_week, is_weekend]
    features_array = np.array(features).reshape(1, -1)
    pred = xbg.predict(features_array)
    st.write(f"Sales Revenue: {pred[0]:.1f}")
    # st.write("### Data received by function:")
    # st.write(f"Latitude: {lat:.6f}")
    # st.write(f"Longitude: {lon:.6f}")
    # st.write(f"Option selected: {option}")
    # st.write(f"Date selected: {[day,month,year]}")

m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
m.add_child(folium.LatLngPopup())

st_data = st_folium(m, width=700, height=500)

lat = lon = selected_option = selected_date = None

if st_data and st_data['last_clicked']:
    lat = st_data['last_clicked']['lat']
    lon = st_data['last_clicked']['lng']
    st.success(f"Clicked location: Latitude: {lat:.6f}, Longitude: {lon:.6f}")

    options = [
        'Bed', 'Bluetooth Speaker', 'Board Game', 'Camera', 'Cereal', 'Coffee', 'Cookbook',
 'Dining Table', 'Doll', 'Energy Drinks', 'Formal Shirt', 'Gaming Console', 'Headphones',
 'Jeans', 'Laptop', 'Men T-shirt', 'Microwave', 'Novel', 'Organic Snacks', 'Protein Bars',
 'Puzzle', 'Remote Control Car', 'Running Shoes', 'Self-help Book', 'Smartphone',
 'Smartwatch', 'Sofa', 'Sunglasses', 'Sweaters', 'Tablet', 'Textbook', 'Toy Car',
 'Washing Machine', 'Winter Jacket', 'Women Dress'
]
    selected_option = st.selectbox("Select a product option:", options)

    selected_date = st.date_input("Select a date:", date.today())

    if st.button("Submit"):
        process_data(lat, lon, selected_option,selected_date)
