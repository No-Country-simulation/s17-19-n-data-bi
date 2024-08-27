import streamlit as st
import torch
from models.inference import load_model, predict

def stock_verification():
    st.title("Verificación de Stock en Sucursales")
    model = load_model("stock")

    if model:
        # Definir un formulario con un botón de submit
        with st.form(key='stock_form'):
            id_sucursal = st.text_input("Ingrese el ID de la sucursal")
            skuagr_2 = st.text_input("Ingrese el SKU del producto")

            # Botón para enviar el formulario
            submit_button = st.form_submit_button(label='Verificar Stock')

        # Solo ejecutar la predicción si se presiona el botón de submit
        if submit_button:
            try:
                input_data = torch.tensor([float(id_sucursal), float(skuagr_2)])
                result = predict(model, input_data)
                st.write(f'Resultado de la petición: {result}')
            except ValueError:
                st.error("Por favor ingrese valores numéricos válidos.")
            except Exception as e:
                st.error(f"Error al verificar stock: {e}")
