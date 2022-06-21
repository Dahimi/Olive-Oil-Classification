import streamlit as st
import pandas as pd
import plotly_express as px
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from models import mainModel
from prediction import main

print("****************************** Aide ********************************************* ")
print("****************************** Aide ********************************************* ")
print("Si le serveur de la page web ne s'est pas lancé")
print(" c'est à dire, vous ne pouvez voir : Local URL: http://localhost:8501 dans cmd")
print("Alors c'est probablement c'est parce que vous avez exécuté la commande:")
print("python .\Programme_Principal.py")
print("au lieu de la commande : ")
print("streamlit run .\Programme_Principal.py ")
print("donc veuillez exécuté : 'streamlit run .\Programme_Principal.py'")


clf = RandomForestClassifier(n_estimators=500)
loc = Nominatim(user_agent="GetLoc")

st.sidebar.subheader("Visualization Settings")
uploaded_file = st.sidebar.file_uploader(label="To work with another DataSet please upload your CSV or Excel file", type=['csv', 'xlsx'])
df = pd.read_csv("test3.csv")
df1 = pd.read_csv("test3.csv")
df1['location'] = df1['Region'] + ';' + df1['Area']
X = df1.drop(['Region', 'Area', 'location'], axis=1)
y = df1['location']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
clf.fit(X_train, y_train)

def show():
    st.header("Exploratory Data Analysis")
    st.write("Before getting into the **machine learning part ** we need first to do some data cleaning and exploration,to gain insights from the data and make sure that the it is well suited for our models.")
    st.write("This could be done through the following steps : ")
    st.write("* Data Loading")
    st.write("* Data Cleaning")
    st.write("* Extracting statistics from the dataset")
    st.write("* Exploratory analysis and visualizations")
    st.write("Since the data is already cleaned we will ignore the second step")
    st.write("Exploratory data visualizations (EDVs) are the type of visualizations we assemble when we do not have a clue about what information lies within our dataset.")

def show_data():
    global df
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            df = pd.read_excel(uploaded_file)
    global numeric_columns
    global char_columns
    try:
        numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
        char_columns = list(df.select_dtypes(['object']).columns)
        shown1 = st.sidebar.checkbox("Show Data Set")

        if shown1:
            st.subheader("Show Dataset")
            st.write(df)
            firstEda()
        show_graphs()
    except:
        st.write("Please upload your data set")

def firstEda():
    df["Location"] = df["Region"] + ';' + df["Area"]

    uniqueRegions = df.Region.unique()
    dic = {}

    @st.cache
    def coordonates():
        for region in uniqueRegions:
            if  region!="Inland Sardinia":
                dic[region] = (loc.geocode(region).latitude, loc.geocode(region).longitude)
            else:
                dic[region] = (40.078072,9.283447)
        df["latitude"] = df.Region.map(lambda region: dic.get(region)[0])
        df["longitude"] = df.Region.map(lambda region: dic.get(region)[1])
    coordonates()
    st.subheader("Extracting statistics from the dataset")
    st.write("The table below describes the kay statistical features of the dataset in concise way -from the mean , to the max and min value of each numerical column")
    st.write(df.describe())
    st.write("**Let's see how the regions are distributed in the dataset**")
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(x="Region", data=df)

    st.pyplot(fig)
    st.write("It is clear from the bar-chart above, we can see that :")

    def getPercentage(region):
        per = len(df.loc[df.Region == region]) / len(df) * 100
        return "_  {} : {} %_  ".format(region,round(per))
    comment = ""
    for region in uniqueRegions:
        comment += getPercentage(region)
    st.write(comment)
    st.subheader("Data Insights and Correlation between Columns")
    st.write('''The goal of this part is to understand the relationship between different attributes in the dataset. Since the labels - **Region and Area ** - are categorical and non-numerical
             we need first to find a numerical representation of of them. And the best way, is to use the Latitude and Longitude which can be done with the following code snippet''')
    with st.echo():
        uniqueRegions = df.Region.unique()
        dic = {}

        @st.cache
        def coordonates():
            for region in uniqueRegions:
                if region != "Inland Sardinia":
                    dic[region] = (loc.geocode(region).latitude, loc.geocode(region).longitude)
                else:
                    dic[region] = (40.078072, 9.283447)
            df["latitude"] = df.Region.map(lambda region: dic.get(region)[0])
            df["longitude"] = df.Region.map(lambda region: dic.get(region)[1])

        coordonates()
    st.write("Now we could calculate the correlation matrix of the dataset, therefore we can get the following visualizations :")
    numericDf = df.select_dtypes('number')
    corr = numericDf.corr()
    st.write("**Bar Charts**")
    fig2, (ax1, ax2) = plt.subplots(figsize=(35, 18), nrows=1, ncols=2)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontsize(30)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontsize(30)
    ax1.set_title("Correlation with latitude")
    ax2.set_title("Correlation with longitude")
    lat_corr = corr['latitude'].drop(['latitude', 'longitude'])
    lat_corr.plot(kind='bar', colormap='turbo', ax=ax1)
    lon_corr = corr['longitude'].drop(['latitude', 'longitude'])
    lon_corr.plot.bar(ax=ax2)
    fig2.tight_layout(pad=3.0)
    st.pyplot(fig2)
    st.write("**Heatmap**")
    fig=plt.figure()
    plt.title("Correlation between all attributres")
    sns.heatmap(corr, annot=True, cmap='cool')
    st.pyplot(fig)
    st.write("We can conclude from the visualizations above that, the features that have a big impact on the region of __ an olive oil__ are (by order of importance): ")
    st.write("1. Linolenic")
    st.write("2. Oleic")
    st.write("3. Arachidic")

def show_graphs():
    shown2 = st.sidebar.checkbox("Show Graphs")
    if uploaded_file is not None or shown2:
        if shown2:
            st.sidebar.subheader("Graph Settings")
            try:
                st.subheader("Statistical Analysis for each feature with respect to the Region or Area")
                st.write("The charts below, are box-plots that give an overview of each feature with respect to the Region or Area ( This includes: min, max, mean, std, outliers, etc.")
                x_values = st.sidebar.selectbox('Caracteristique', options=numeric_columns)
                y_values = st.sidebar.selectbox('Region', options=char_columns)
                plot = px.box(data_frame=df, y=x_values, x=y_values)
                st.plotly_chart(plot)

            except Exception as e:
                print(e)
def head():
    col1, col2, col3 = st.beta_columns([1, 1, 1])
    with col1:
        st.title("Olive Oil Classification")
    with col3:
        image = Image.open('sunrise.png')
        st.image(image, caption='Created by Soufiane & Hamza')
head()
rad=st.sidebar.radio("Navigation",["Exploratory Data Analysis","Model Study","Predictions"])
if rad=="Exploratory Data Analysis":
    show()
    show_data()

elif rad == 'Model Study':
    clf = mainModel(df)
elif rad == "Predictions":
    main(clf)