import torch
import streamlit as st
from .inference import load_model, predict  # Adjust the import based on your directory structure

def stock_verification():
    st.title("Verificación de Stock en Sucursales")
    model = load_model("stock")

    if model:
        if 'sucursal_id' not in st.session_state:
            st.session_state['sucursal_id'] = ''
        if 'skuagr_2' not in st.session_state:
            st.session_state['skuagr_2'] = ''


        sucursal_id = st.text_input("Ingrese el ID de la sucursal", value=st.session_state['sucursal_id'])
        skuagr_2 = st.text_input("Ingrese el SKU del producto", value=st.session_state['skuagr_2'])

        if st.button('Verificar Stock'):
            st.session_state['sucursal_id'] = sucursal_id
            st.session_state['skuagr_2'] = skuagr_2

            try:
                input_data = torch.tensor([float(sucursal_id), float(skuagr_2)])
                result = predict(model, input_data)
                st.write(f'Resultado de la predicción: {result}')
            except ValueError:
                st.error("Por favor ingrese valores numéricos válidos.")
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")
