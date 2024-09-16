import pandas as pd
import streamlit as st
import torch
from models.inference import load_model, predict
from models.stock_data import load_stock_data

def preprocess_input_data(stock_data, id_sucursal, skuagr_2):
    # Convertir los valores de entrada a los tipos correctos
    id_sucursal = int(id_sucursal.strip())  # Aseguramos que sea un entero y sin espacios
    skuagr_2 = skuagr_2.strip()  # Eliminamos posibles espacios en blanco

    # Filtrar los datos correctamente utilizando el DataFrame cargado
    filtered_data = stock_data[
        (stock_data['id_sucursal'] == id_sucursal) & 
        (stock_data['skuagr_2'] == skuagr_2)
    ]
    
    # Debugging: Mostrar los datos filtrados
    st.write("Datos filtrados:", filtered_data)

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

    # Verificar si input_data está vacío antes de convertirlo en tensor
    if input_data.empty:
        raise ValueError("Los datos de entrada están vacíos, no se puede realizar la predicción.")
    
    # Convertir input_data en tensor
    input_tensor = torch.tensor(input_data.values).float()

    # Realizar la predicción usando el modelo
    prediction = predict(model, input_tensor)
    
    # Mostrar los resultados
    st.write(f"Predicción de stock para SKU {skuagr_2} en sucursal {id_sucursal} dentro de los próximos 30 días de acuerdo a la periodicidad de venta, probabilidad de (* 100): {prediction.item()}")

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
