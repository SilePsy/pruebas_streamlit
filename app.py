# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:08:59 2024

@author: johnf
"""

# Importar las bibliotecas necesarias
import streamlit as st
import numpy as np
import pickle

# Cargar el modelo de clasificación entrenado
with open("modelo_clasificacion.pkl", "rb") as f:
    model = pickle.load(f)

# Título de la aplicación
st.title("Aplicación Web de Clasificación con Regresión Logística")

st.write("""
### Introduce los valores para las características del modelo:
""")

# Crear entradas de usuario para cada una de las 10 características
feature_1 = st.number_input("Característica 1", value=0.0)
feature_2 = st.number_input("Característica 2", value=0.0)
feature_3 = st.number_input("Característica 3", value=0.0)
feature_4 = st.number_input("Característica 4", value=0.0)
feature_5 = st.number_input("Característica 5", value=0.0)
feature_6 = st.number_input("Característica 6", value=0.0)
feature_7 = st.number_input("Característica 7", value=0.0)
feature_8 = st.number_input("Característica 8", value=0.0)
feature_9 = st.number_input("Característica 9", value=0.0)
feature_10 = st.number_input("Característica 10", value=0.0)

# Convertir los valores de entrada en un array numpy
input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5,
                        feature_6, feature_7, feature_8, feature_9, feature_10]])

# Cuando el usuario presiona el botón "Predecir"
if st.button("Predecir"):
    # Hacer predicción
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Mostrar los resultados
    st.write(f"Predicción de la clase: **{int(prediction[0])}**")
    st.write(f"Probabilidades de predicción: Clase 0: {prediction_proba[0][0]:.4f}, Clase 1: {prediction_proba[0][1]:.4f}")
