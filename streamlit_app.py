# Código en proceso aún...

import streamlit as st
import pandas as pd

# Cargar datos directamente
@st.cache_data
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

    stock_actualizado['id_sucursal'] = stock_actualizado['id_sucursal'].astype(str)
    stock_actualizado['skuagr_2'] = stock_actualizado['skuagr_2'].astype(str)

    return stock_actualizado

# Mostrar resultados
def show_stock_result(stock_data, id_sucursal, skuagr_2):
    filtered_data = stock_data[
        (stock_data['id_sucursal'] == id_sucursal) & 
        (stock_data['skuagr_2'] == skuagr_2)
    ]

    if not filtered_data.empty:
        st.write("Resultados de la consulta:")
        st.write(filtered_data)
    else:
        st.write("No se encontraron registros para la sucursal y SKU proporcionados.")

# Lógica principal de verificación de stock
def stock_verification():
    st.title("Verificación de Stock en Sucursales")
    stock_data = load_stock_data()

    with st.form(key='stock_form'):
        id_sucursal = st.text_input("Ingrese el ID de la sucursal")
        skuagr_2 = st.text_input("Ingrese el SKU del producto")

        submit_button = st.form_submit_button(label='Verificar Stock')

    if submit_button:
        show_stock_result(stock_data, id_sucursal, skuagr_2)

# Interfaz principal
if __name__ == "__main__":
    st.sidebar.image('streamlit_app/Pi.png', use_column_width=True)
    st.sidebar.title("Bienvenid@! Selecciona el insight:")

    gestion_stocks = st.sidebar.button('GESTIÓN DE STOCKS')

    if gestion_stocks:
        stock_verification()
