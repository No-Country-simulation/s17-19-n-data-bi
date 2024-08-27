import streamlit as st
import torch
from models.inference import load_model, predict
from models.stock_result import load_stock_data

def stock_verification():
    st.title("Verificación de Stock en Sucursales")

    model = load_model("stock")
    
    # Paso 1: Verificar la captura de datos del formulario
    if model:
        with st.form(key='stock_form'):
            id_sucursal = st.text_input("Ingrese el ID de la sucursal")
            skuagr_2 = st.text_input("Ingrese el SKU del producto")
            submit_button = st.form_submit_button(label='Verificar Stock')
        
        if submit_button:
            st.write(f"Datos ingresados: ID Sucursal: {id_sucursal}, SKU: {skuagr_2}")
            
            # Paso 2: Verificar la carga de datos
            try:
                stock_data = load_stock_data()
                st.write("Datos de stock cargados correctamente.")
            except Exception as e:
                st.error(f"Error al cargar los datos de stock: {e}")
                return
            
            # Paso 3: Verificar el filtrado de datos
            try:
                id_sucursal = int(id_sucursal)  # Asegurarnos de que sea un número
                resultado = stock_data[(stock_data['id_sucursal'] == id_sucursal) & 
                                       (stock_data['skuagr_2'] == skuagr_2)]
                
                if resultado.empty:
                    st.warning("No se encontraron registros para los valores ingresados.")
                else:
                    st.write("Resultado encontrado:")
                    st.write(resultado)
            except ValueError:
                st.error("Por favor ingrese valores numéricos válidos.")
            except Exception as e:
                st.error(f"Error al verificar stock: {e}")


