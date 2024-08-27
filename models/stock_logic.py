import streamlit as st
import torch
from models.inference import load_model, predict
from models.stock_result import load_stock_data

def stock_verification():
    st.title("Verificación de Stock en Sucursales")
    
    # Cargar los datos de stock
    try:
        stock_data = load_stock_data()
        st.write("Datos de stock cargados correctamente.")
    except Exception as e:
        st.error(f"Error al cargar los datos de stock: {e}")
        return
    
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
            st.write(f"ID Sucursal: {id_sucursal}, SKU: {skuagr_2}")
            try:
                # Convertir id_sucursal a entero para la comparación
                id_sucursal = int(id_sucursal)

                # Filtrar los datos de stock para la sucursal y SKU ingresados
                resultado = stock_data[(stock_data['id_sucursal'] == id_sucursal) & 
                                       (stock_data['skuagr_2'] == skuagr_2)]
                
                if resultado.empty:
                    st.warning("No se encontraron registros para los valores ingresados.")
                else:
                    # Mostrar el resultado en Streamlit
                    st.write("Resultado encontrado:")
                    st.write(resultado)
            except ValueError:
                st.error("Por favor ingrese valores numéricos válidos.")
            except Exception as e:
                st.error(f"Error al verificar stock: {e}")

