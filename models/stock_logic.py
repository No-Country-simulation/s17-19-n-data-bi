import streamlit as st
import torch
from models.inference import load_model, predict
from models.stock_data import load_stock_data

def show_stock_result(stock_data, id_sucursal, skuagr_2):
    # Filtrar los datos por los inputs del usuario
    filtered_data = stock_data[
        (stock_data['id_sucursal'] == id_sucursal) & 
        (stock_data['skuagr_2'] == skuagr_2)
    ]

    # Mostrar los datos filtrados para depuración
    st.write("Datos filtrados:")
    st.write(filtered_data)

    # Mostrar el resultado o un mensaje si no se encuentra nada
    if not filtered_data.empty:
        st.write("Resultados de la consulta:")
        st.write(filtered_data)
    else:
        st.write("No se encontraron registros para la sucursal y SKU proporcionados.")

def stock_verification():
    # Cargar los datos una vez
    stock_data = load_stock_data()

    # Mostrar los datos cargados para depuración
    st.write("Disponibilidad de Datos Actuales:")
    st.write(stock_data)

    st.title("Aplicar Filtro de Verificación de Stock en Sucursales")

    # Crear el formulario para ingresar los datos
    with st.form(key='stock_form'):
        id_sucursal = st.text_input("Ingrese el ID de la sucursal")
        skuagr_2 = st.text_input("Ingrese el SKU del producto")

        # Botón para enviar el formulario
        submit_button = st.form_submit_button(label='Verificar Stock')

    
    # Verificar si se han ingresado datos válidos y mostrar resultados
    if submit_button:
        if id_sucursal and skuagr_2:
            # Filtrar los datos por los inputs del usuario
            filtered_data = stock_data[
                (stock_data['id_sucursal'] == id_sucursal) & 
                (stock_data['skuagr_2'] == skuagr_2)
            ]

            # Guardar los resultados en session state
            st.session_state['filtered_data'] = filtered_data

    # Mostrar los resultados si están disponibles
    if st.session_state['filtered_data'] is not None:
        if not st.session_state['filtered_data'].empty:
            st.write("Resultados de la consulta:")
            st.write(st.session_state['filtered_data'])
        else:
            st.write("No se encontraron registros para la sucursal y SKU proporcionados.")
