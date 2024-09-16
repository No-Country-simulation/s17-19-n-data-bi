
import pandas as pd
import streamlit as st
import torch
from models.inference import load_model, predict
from models.stock_data import load_stock_data

def preprocess_input_data(stock_data, id_sucursal, skuagr_2):
    # Filtrar los datos correctamente utilizando el DataFrame cargado
    filtered_data = stock_data[
        (stock_data['id_sucursal'] == id_sucursal) & 
        (stock_data['skuagr_2'] == skuagr_2)
    ]

    # Verificar si existen los datos filtrados
    if filtered_data.empty:
        raise ValueError(f"No se encontraron datos para id_sucursal {id_sucursal} y skuagr_2 {skuagr_2}")
    
    # Usar las columnas calculadas en 'stock_data.py' como 'stock_disponible' y 'hay_stock'
    input_data = filtered_data[['stock_disponible', 'hay_stock']]

    # Crear variables dummy para coincidir con el preprocesamiento del modelo
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    # Asegurarse de que el número de columnas coincida con lo esperado por el modelo
    if input_data.shape[1] < 1995:  # Asumimos que el modelo espera 1995 características
        missing_cols = 1995 - input_data.shape[1]
        # Añadir columnas adicionales llenas de ceros para completar
        for i in range(missing_cols):
            input_data[f'dummy_{i}'] = 0

    return input_data

def show_stock_result(stock_data, id_sucursal, skuagr_2, model):
    # Preprocesar los datos para la inferencia
    input_data = preprocess_input_data(stock_data, id_sucursal, skuagr_2)

    # Verificar que las columnas de entrada coincidan con lo que espera el modelo
    expected_columns = model.fc1.in_features
    if input_data.shape[1] != expected_columns:
        raise ValueError(f"El número de características de input_data ({input_data.shape[1]}) no coincide con el esperado por el modelo ({expected_columns})")

    # Realizar la predicción usando el modelo
    prediction = predict(model, torch.tensor(input_data.values).float())
    
    # Mostrar los resultados
    st.write(f"Predicción de stock para SKU {skuagr_2} en sucursal {id_sucursal}: {prediction.item()}")
