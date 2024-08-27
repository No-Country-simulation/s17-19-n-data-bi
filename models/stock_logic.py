import streamlit as st
import torch
import pandas as pd
from models.inference import load_model, predict

def stock_verification():
    st.title("Verificación de Stock en Sucursales")
    model = load_model("stock")

    if model:
        # Definir un formulario con un botón de submit
        with st.form(key='stock_form'):
            id_sucursal = st.text_input("Ingrese el ID de la sucursal", value="1001")
            skuagr_2 = st.text_input("Ingrese el SKU del producto", value="100094545G5")

            # Botón para enviar el formulario
            submit_button = st.form_submit_button(label='Verificar Stock')

        # Solo ejecutar la predicción si se presiona el botón de submit
        if submit_button:
            try:
                # Convertir id_sucursal a número entero (sin comas)
                id_sucursal = id_sucursal.replace(",", "")
                # Convertir datos a flotantes para el modelo
                input_data = torch.tensor([float(id_sucursal), float(skuagr_2)])  
                result = predict(model, input_data)
                
                # Cargar y filtrar los datos de stock
                stock_data = load_stock_data()  # Esta función se llama aquí
                filtered_data = stock_data[
                    (stock_data['id_sucursal'] == int(id_sucursal)) & 
                    (stock_data['skuagr_2'] == skuagr_2)
                ]
                
                if not filtered_data.empty:
                    st.write("Resultados de la consulta:")
                    st.write(filtered_data)
                else:
                    st.write("No se encontraron registros para la sucursal y SKU proporcionados.")

            except ValueError:
                st.error("Por favor ingrese valores numéricos válidos para la sucursal y SKU.")
            except Exception as e:
                st.error(f"Error al verificar stock: {e}")

def load_stock_data():
    sucursales = pd.read_parquet('data/Sucursales.parquet')
    productos = pd.read_parquet('data/Productos.parquet')
    data = pd.read_parquet('data/Data.parquet')

    stock_inicial = pd.merge(sucursales[['id_sucursal']], productos[['skuagr_2']], how='cross')
    stock_inicial['stock_inicial'] = 100  # Asignar 100 unidades como stock inicial

    transacciones_agrupadas = data.groupby(['id_sucursal', 'skuagr_2']).agg({'cantidad_dispensada': 'sum'}).reset_index()
    stock_actualizado = pd.merge(stock_inicial, transacciones_agrupadas, on=['id_sucursal', 'skuagr_2'], how='left')
    stock_actualizado['cantidad_dispensada'] = stock_actualizado['cantidad_dispensada'].fillna(0)
    stock_actualizado['stock_disponible'] = stock_actualizado['stock_inicial'] - stock_actualizado['cantidad_dispensada']
    stock_actualizado['hay_stock'] = stock_actualizado['stock_disponible'] > 0
    stock_actualizado['hay_stock'] = stock_actualizado['hay_stock'].astype(int)

    return stock_actualizado

