
import pandas as pd
import streamlit as st
import torch
from models.inference import load_model, predict
from models.stock_data import load_stock_data

def preprocess_input_data(stock_data, id_sucursal, skuagr_2):
    stock_inicial = pd.DataFrame({"id_sucursal": [id_sucursal], "skuagr_2": [skuagr_2]})
    stock_inicial["stock_inicial"] = 300
    stock_inicial["cantidad_dispensada"] = 0
    stock_inicial["stock_disponible"] = stock_inicial["stock_inicial"] - stock_inicial["cantidad_dispensada"]
    input_data = stock_inicial.drop(columns=["stock_inicial", "cantidad_dispensada", "stock_disponible"])
    input_data = pd.get_dummies(input_data, drop_first=True)
    return input_data

def show_stock_result(stock_data, id_sucursal, skuagr_2, model):
    input_data = preprocess_input_data(stock_data, id_sucursal, skuagr_2)
    if input_data.empty:
        st.warning("No se encontraron registros para la sucursal y SKU proporcionados.")
        return

    expected_columns = model.fc1.in_features
    if input_data.shape[1] != expected_columns:
        raise ValueError(f"El número de características de input_data ({input_data.shape[1]}) no coincide con lo esperado ({expected_columns}).")

    with torch.no_grad():
        prediction = predict(model, input_data)

    st.write(f"Predicción de disponibilidad de stock: {'Disponible' if prediction else 'No Disponible'}")

def stock_verification():
    stock_data = load_stock_data()
    model = load_model('stock')

    with st.form(key='stock_form'):
        id_sucursal = st.text_input("Ingrese el ID de la sucursal", key="id_sucursal")
        skuagr_2 = st.text_input("Ingrese el SKU del producto", key="skuagr_2")
        submit_button = st.form_submit_button(label='Verificar Stock')

    if submit_button:
        if id_sucursal and skuagr_2:
            st.write("Verificando stock para Sucursal:", id_sucursal, "y SKU:", skuagr_2)
            show_stock_result(stock_data, id_sucursal, skuagr_2, model)
        else:
            st.warning("Por favor, ingrese tanto el ID de la sucursal como el SKU del producto.")
