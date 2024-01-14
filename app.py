
import pandas as pd
import numpy as np
import xgboost
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, KFold
import warnings
import pickle
import streamlit as st
import re
import statistics
st.set_page_config(layout="wide")

st.title("INDUSTRIAL COPPER MODELLING")

tab1, tab2 = st.tabs(["ABOUT", "PREDICT"])

with tab1:

        st.markdown("# :blue[Singapore Resale Flat Prices Prediction]")
        st.markdown("### :blue[Overview :] This project aims to construct a machine learning model to predict resale price of the flat. "
                                          "This predictive model is based on historical data of resale flat transactions from 1990 to 2017 onwards, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.")
        st.markdown("### :blue[Model :] XG Boost Regressor")
        st.markdown("### :blue[Features :] Town, Flat type, Flat model, Storey range, Floor area, leace Commence date(year)")
        st.markdown("### :blue[Domain :] Real Estate")

with tab2:


        # Define the possible values for the dropdown menus
        town_options = ['SENGKANG','WOODLANDS','PUNGGOL','JURONG WEST','TAMPINES','YISHUN','BEDOK','HOUGANG','CHOA CHU KANG','ANG MO KIO','BUKIT MERAH','BUKIT PANJANG','BUKIT BATOK','TOA PAYOH',
        'PASIR RIS','KALLANG/WHAMPOA','QUEENSTOWN','SEMBAWANG','GEYLANG','CLEMENTI','JURONG EAST','SERANGOON','BISHAN','CENTRAL AREA','MARINE PARADE','BUKIT TIMAH']
        flat_model_options = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified','Premium Apartment', 'Maisonette', 'Apartment', 'Model A2','Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
        'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation','Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen']
        flat_type_options = ['3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', '1 ROOM','MULTI-GENERATION']

        # Define the widgets for user input
        with st.form("my_form"):
            col1,col2,col3=st.columns([5,2,5])
            with col1:
                st.write(' ')
                town = st.selectbox("Town", town_options,key=1)
                flat_model = st.selectbox("Flat Model", flat_model_options,key=2)
                flat_type = st.selectbox("Flat Type", flat_type_options,key=3)

            with col3:
                floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
                lease_commence_year = st.number_input(('Lease Commence Year'), min_value = 1966, max_value = 2022)
                storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")
                submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")
                st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #009999;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)

            flag=0
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [floor_area_sqm,lease_commence_year]:
                if re.match(pattern, str(i)):
                    pass
                else:
                    flag=1
                    break

        if submit_button and flag==1:
            if len(i)==0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ",i)

        if submit_button and flag==0:

            import pickle

            try:

                with open(r"C:/Users/91890/OneDrive/Desktop/GUVI_DS/Projects/Project 6_Resale price prediction for flats/Updated pickle files/xgbmodel.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open(r'C:/Users/91890/OneDrive/Desktop/GUVI_DS/Projects/Project 6_Resale price prediction for flats/Updated pickle files/scaler.pkl', 'rb') as f:
                    scaler_loaded = pickle.load(f)
                with open(r"C:/Users/91890/OneDrive/Desktop/GUVI_DS/Projects/Project 6_Resale price prediction for flats/Updated pickle files/lc_town.pkl", 'rb') as f:
                    lc_town = pickle.load(f)
                with open(r"C:/Users/91890/OneDrive/Desktop/GUVI_DS/Projects/Project 6_Resale price prediction for flats/Updated pickle files/lc_model.pkl", 'rb') as f:
                    lc_model = pickle.load(f)
                with open(r"C:/Users/91890/OneDrive/Desktop/GUVI_DS/Projects/Project 6_Resale price prediction for flats/Updated pickle files/lc_type.pkl", 'rb') as f:
                    lc_type = pickle.load(f)

                print("Models loaded successfully.")

                split_list = re.split(r'\sTO\s', storey_range)
                float_list = [float(i) for i in split_list]
                storey_median = statistics.median(float_list)
                new_sample = np.array([[town,flat_type,flat_model,int(lease_commence_year),np.log(float(floor_area_sqm)),float(storey_median)]])
                new_sample[:, 0] = lc_town.transform(new_sample[:, 0])
                new_sample[:, 1] = lc_type.transform(new_sample[:, 1])
                new_sample[:, 2] = lc_model.transform(new_sample[:, 2])
                final_sample = scaler_loaded.transform(new_sample)
                new_pred = loaded_model.predict(final_sample)[0]
                print('Predicted selling price:', np.exp(new_pred))
                st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

            except Exception as e:
                print(f"Error loading pickle file: {e}")



