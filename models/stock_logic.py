
import pandas as pd
import streamlit as st
import torch
from models.inference import load_model, predict
from models.stock_data import load_stock_data

def preprocess_input_data(stock_data, id_sucursal, skuagr_2):
    # Generar datos de ejemplo basados en la estructura esperada
    stock_inicial = pd.DataFrame({"id_sucursal": [id_sucursal], "skuagr_2": [skuagr_2]})
    stock_inicial["stock_inicial"] = 300
    stock_inicial["cantidad_dispensada"] = 0
    stock_inicial["stock_disponible"] = stock_inicial["stock_inicial"] - stock_inicial["cantidad_dispensada"]
    
    # Crear variables dummy para coincidir con el preprocesamiento del entrenamiento
    input_data = stock_inicial.drop(columns=["stock_inicial", "cantidad_dispensada", "stock_disponible"])
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    # Asegurar que el número de columnas coincida con lo esperado por el modelo
    if input_data.shape[1] < 1995:  # Asumimos que el modelo espera 1995 características
        missing_cols = 1995 - input_data.shape[1]
        # Añadir columnas adicionales llenas de ceros para completar
        for i in range(missing_cols):
            input_data[f'dummy_{i}'] = 0
    
    return input_data

def show_stock_result(stock_data, id_sucursal, skuagr_2, model):
    # Filtrar los datos correctamente utilizando el DataFrame cargado
    filtered_data = stock_data[
        (stock_data['id_sucursal'] == id_sucursal) & 
        (stock_data['skuagr_2'] == skuagr_2)
    ]

    # Mostrar el resultado o un mensaje si no se encuentra nada
    if not filtered_data.empty:
        st.write("Resultados de la consulta:")
        st.write(filtered_data)
        
        # Preprocesar los datos para la inferencia
        input_data = preprocess_input_data(stock_data, id_sucursal, skuagr_2)

        expected_columns = model.fc1.in_features
        if input_data.shape[1] != expected_columns:
            raise ValueError(f"El número de características de input_data ({input_data.shape[1]}) no coincide con lo esperado ({expected_columns}).")
        
        # Convertir a tensores y hacer la predicción
        with torch.no_grad():
            prediction = predict(model, input_data)

        # Mostrar la predicción
        st.write(f"Predicción de disponibilidad de stock: {'Disponible' if prediction else 'No Disponible'}")
    else:
        st.warning("No se encontraron registros para la sucursal y SKU proporcionados.")

def stock_verification():
    # Cargar los datos una vez
    stock_data = load_stock_data()

    # Cargar el modelo de inferencia
    model = load_model('stock')

    # Crear el formulario para ingresar los datos
    with st.form(key='stock_form'):
        id_sucursal = st.text_input("Ingrese el ID de la sucursal", key="id_sucursal")
        skuagr_2 = st.text_input("Ingrese el SKU del producto", key="skuagr_2")

        # Botón para enviar el formulario
        submit_button = st.form_submit_button(label='Verificar Stock')

    # Verificar si se han ingresado datos válidos y mostrar resultados
    if submit_button:
        if id_sucursal and skuagr_2:
            st.write("Verificando stock para Sucursal:", id_sucursal, "y SKU:", skuagr_2)
            show_stock_result(stock_data, id_sucursal, skuagr_2, model)
        else:
            st.warning("Por favor, ingrese tanto el ID de la sucursal como el SKU del producto.")
