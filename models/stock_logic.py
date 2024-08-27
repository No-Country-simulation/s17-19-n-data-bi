import streamlit as st
import torch
from models.inference import load_model, predict
from models.stock_data import load_stock_data

def show_stock_result(stock_data, id_sucursal, skuagr_2, model):
    # Filtrar los datos por los inputs del usuario
    filtered_data = stock_data[
        (stock_data['id_sucursal'] == id_sucursal) & 
        (stock_data['skuagr_2'] == skuagr_2)
    ]

    # Mostrar el resultado o un mensaje si no se encuentra nada
    if not filtered_data.empty:
        st.write("Resultados de la consulta:")
        st.write(filtered_data)
        
        # Preparar los datos para la inferencia
        input_data = filtered_data.drop(columns=['stock_inicial', 'cantidad_dispensada', 'stock_disponible', 'hay_stock'])

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
            st.write("Verificando stock para Sucursal:", id_sucursal, "y SKU:", skuagr_2)  # Depuración

            # Mostrar los datos filtrados usando la función show_stock_result
            show_stock_result(stock_data, id_sucursal, skuagr_2, model)
        else:
            st.warning("Por favor, ingrese tanto el ID de la sucursal como el SKU del producto.")

