import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
import folium
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from geopy.geocoders import Nominatim
loc = Nominatim(user_agent="GetLoc")


def main(clf):
    st.header("Predictions")
    st.subheader("Form")
    with st.form(key='salaryform'):
        col1, col2 = st.beta_columns([1, 1])
        with col1:
            palmitic = st.number_input("palmitic")
        with col2:
            palmitoleic = st.number_input("palmitoleic")
        col3, col4 = st.beta_columns([1, 1])
        with col3:
            stearic = st.number_input("stearic	")
        with col4:
            oleic = st.number_input("oleic")
        col5, col6 = st.beta_columns([1, 1])
        with col5:
            linoleic = st.number_input("linoleic ")
        with col6:
            linolenic = st.number_input("linolenic ")
        col7, col8 = st.beta_columns([1, 1])
        with col7:
            arachidic = st.number_input("arachidic ")
        with col8:
            eicosenoic = st.number_input("eicosenoic ")
        col9, col10, col11 = st.beta_columns([3, 3, 1])
        with col10:
            predict = st.form_submit_button(label='Predict')

    if predict:

        list=[[palmitic,palmitoleic,stearic,oleic,linoleic,linolenic,arachidic,eicosenoic]]
        y_pred = clf.predict(list)[0]
        region, area = y_pred.split(";")[0], y_pred.split(";")[1]
        dic={}
        if region != "Inland Sardinia":
            dic[region] = (loc.geocode(region).latitude, loc.geocode(region).longitude)
        else:
            dic[region] = (40.078072, 9.283447)
        st.subheader("Map")
        m = folium.Map(location=[dic[region][0],dic[region][1]], zoom_start=6)
        tooltip = region
        folium.Marker([dic[region][0],dic[region][1]], popup=region, tooltip=tooltip).add_to(m)
        folium_static(m)
        st.subheader("Results")
        st.write("This olive oil is located in the region **" + region + "** and the area **" + area + "**")
        with st.beta_expander("Results"):
            df = pd.DataFrame({'palmitic': [palmitic], 'palmitoleic': [palmitoleic], 'stearic': [stearic],
                                   'oleic': [oleic], 'linoleic': [linolenic], 'col6': [linolenic],
                                   'arachidic': [arachidic], 'eicosenoic': [eicosenoic],
                                    'region': [region], 'area': [area]})
            st.dataframe(df.T)


