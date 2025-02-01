import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
filename='mpdel.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)
st.title('Survavied_Preadction')
st.subheader('Please enter your data:')

df = pd.read_csv('passenger.csv')
columns_list = df.columns.to_list()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    object_columns = df.select_dtypes(include=['object']).columns

    for col in object_columns: 
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
   
    df_preprocessed = df[columns_list].fillna(0)

    prediction = loaded_model.predict(df_preprocessed)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    st.write(prediction_text)

