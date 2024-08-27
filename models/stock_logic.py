import streamlit as st
import torch
from models.inference import load_model, predict
from models.stock_result import load_stock_data

# Cargar el modelo una vez
model = load_model("stock")

# Cargar los datos de stock una vez
try:
    stock_data = load_stock_data()
except Exception as e:
    st.error(f"Error al cargar los datos de stock: {e}")
    stock_data = None

def stock_verification():
    st.title("Verificación de Stock en Sucursales")
    
    if model is None or stock_data is None:
        st.error("No se pudo cargar el modelo o los datos. Por favor, verifica el entorno.")
        return
    
    with st.form(key='stock_form'):
        id_sucursal = st.text_input("Ingrese el ID de la sucursal")
        skuagr_2 = st.text_input("Ingrese el SKU del producto")
        submit_button = st.form_submit_button(label='Verificar Stock')
    
    if submit_button:
        try:
            id_sucursal = int(id_sucursal)
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
