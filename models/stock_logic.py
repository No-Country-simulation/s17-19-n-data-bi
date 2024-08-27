import streamlit as st
import torch
from models.inference import load_model, predict
from models.stock_result import show_stock_result

def stock_verification():
    st.title("Verificación de Stock en Sucursales")
    model = load_model("stock")

    if model:
        with st.form(key='stock_form'):
            id_sucursal = st.text_input("Ingrese el ID de la sucursal")
            skuagr_2 = st.text_input("Ingrese el SKU del producto")

            submit_button = st.form_submit_button(label='Verificar Stock')

        if submit_button:
            try:
                id_sucursal = id_sucursal.replace(",", "").strip()
                skuagr_2 = skuagr_2.strip()

                if not id_sucursal.isdigit():
                    st.error("El ID de la sucursal debe ser un número entero.")
                    return

                input_data = torch.tensor([float(id_sucursal), float(skuagr_2)])
                result = predict(model, input_data)

                # Llamar a stock_result.py para mostrar los resultados
                show_stock_result(id_sucursal, skuagr_2)

            except ValueError:
                st.error("Por favor ingrese valores numéricos válidos para la sucursal y SKU.")
            except Exception as e:
                st.error(f"Error al verificar stock: {e}")


