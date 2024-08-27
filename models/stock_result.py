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

    return stock_actualizado
