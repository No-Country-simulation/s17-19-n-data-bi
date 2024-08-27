import streamlit as st
import pandas as pd

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

    # Asegurarse de que las columnas tienen los tipos correctos
    stock_actualizado['id_sucursal'] = stock_actualizado['id_sucursal'].astype(int)
    stock_actualizado['skuagr_2'] = stock_actualizado['skuagr_2'].astype(str)

    return stock_actualizado

def show_stock_result(id_sucursal, skuagr_2):
    stock_data = load_stock_data()

    # Filtrar los datos según los valores ingresados
    filtered_data = stock_data[
        (stock_data['id_sucursal'] == int(id_sucursal)) & 
        (stock_data['skuagr_2'] == skuagr_2)
    ]

    if not filtered_data.empty:
        st.write("Resultados de la consulta:")
        st.write(filtered_data)
    else:
        st.write("No se encontraron registros para la sucursal y SKU proporcionados.")
