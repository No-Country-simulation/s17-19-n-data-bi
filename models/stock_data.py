import pandas as pd
import numpy as np

def load_stock_data():
    # Cargar datos desde los archivos Parquet
    sucursales = pd.read_parquet('data/Sucursales.parquet')
    productos = pd.read_parquet('data/Productos.parquet')
    data = pd.read_parquet('data/Data.parquet')

    # Crear la tabla de stock inicial y calcular el stock disponible
    stock_inicial = pd.merge(sucursales[['id_sucursal']], productos[['skuagr_2']], how='cross')
    
    # Asignar un stock inicial aleatorio (mayor de 0) a cada combinación de sucursal y producto
    # Usamos randint para generar valores aleatorios entre 0 y 30 (ajustar el rango)
    stock_inicial['stock_inicial'] = np.random.randint(0, 30, size=len(stock_inicial))
    
    # Mostrar una vista previa del stock inicial
    stock_inicial.head()
    
    # 1. Cruzar las transacciones con las sucursales (por id_sucursal)
    # Verificar si la columna 'id_sucursal' está en data, si no, asignar valores aleatorios
    if 'id_sucursal' not in data.columns:
        # Si no existe 'id_sucursal', asignar una sucursal aleatoria
        sucursales_unicas = sucursales['id_sucursal'].unique()
        data['id_sucursal'] = np.random.choice(sucursales_unicas, size=len(data))
    
    # 2. Cruzar las transacciones con los productos (por skuagr_2)
    # Verificar que la columna 'skuagr_2' exista en los tres datasets: data, sucursales, productos
    if 'skuagr_2' in data.columns and 'skuagr_2' in productos.columns:
        # Hacemos el cruce de data con productos
        data_productos = pd.merge(data, productos[['skuagr_2', 'descriprod_agrp2']], on='skuagr_2', how='left')
    else:
        raise KeyError("Error: No se encontró la columna 'skuagr_2' en uno de los datasets.")
    
    # 3. Sumar las transacciones para cada combinación de sucursal y producto
    if 'cantidad_dispensada' not in data_productos.columns:
        # Si no existe la columna, crearla con un valor por defecto (puedes ajustar según tus datos)
        data_productos['cantidad_dispensada'] = np.random.randint(1, 10, size=len(data_productos))  # Ejemplo de valores aleatorios
    
    # Agrupar por sucursal y producto para sumar la cantidad dispensada
    transacciones_agrupadas = data_productos.groupby(['id_sucursal', 'skuagr_2']).agg({'cantidad_dispensada': 'sum'}).reset_index()
    
    # 4. Cruzar transacciones agrupadas con el stock inicial
    # Asegurarse de que 'stock_inicial' tiene las columnas necesarias: 'id_sucursal' y 'skuagr_2'
    if 'id_sucursal' in stock_inicial.columns and 'skuagr_2' in stock_inicial.columns:
        # Cruzamos el stock inicial con las transacciones agrupadas
        stock_actualizado = pd.merge(stock_inicial, transacciones_agrupadas, on=['id_sucursal', 'skuagr_2'], how='left')
        
        # Si hay productos que no se han vendido (no tienen transacciones), llenar con 0
        stock_actualizado['cantidad_dispensada'] = stock_actualizado['cantidad_dispensada'].fillna(0)
    
        # Calcular el stock disponible restando las dispensas al stock inicial
        stock_actualizado['stock_disponible'] = stock_actualizado['stock_inicial'] - stock_actualizado['cantidad_dispensada']
    
        # Calcular la variable objetivo 'hay_stock' (si hay o no stock disponible)
        stock_actualizado['hay_stock'] = (stock_actualizado['stock_disponible'] > 0).astype(int)
    else:
        raise KeyError("Error: 'id_sucursal' o 'skuagr_2' no se encuentran en el dataset 'stock_inicial'.")
    
    # 5. Previsión de stock a 30 días (opcional)
    # Suponiendo que quieres hacer una previsión de la cantidad dispensada en 30 días, usando la tasa diaria promedio
    if 'fecha' in data_productos.columns:
        data_productos['fecha'] = pd.to_datetime(data_productos['fecha'])
        dias_de_ventas = (data_productos['fecha'].max() - data_productos['fecha'].min()).days + 1  # Días de ventas
    
        # Calcular el promedio diario de dispensado por producto y sucursal
        transacciones_agrupadas['promedio_diario'] = transacciones_agrupadas['cantidad_dispensada'] / dias_de_ventas
    
        # Unir la tasa promedio diaria con el stock actualizado
        stock_actualizado = pd.merge(stock_actualizado, transacciones_agrupadas[['id_sucursal', 'skuagr_2', 'promedio_diario']], on=['id_sucursal', 'skuagr_2'], how='left')
    
        # Previsión para 30 días (lo que se dispensaría en 30 días)
        stock_actualizado['prevision_30_dias'] = stock_actualizado['promedio_diario'] * 30
    
        # Calcular el stock futuro: stock disponible menos lo que se espera dispensar en 30 días
        stock_actualizado['stock_futuro'] = stock_actualizado['stock_disponible'] - stock_actualizado['prevision_30_dias']
    
        # Calcular si habrá stock en 30 días
        stock_actualizado['hay_stock_30_dias'] = (stock_actualizado['stock_futuro'] > 0).astype(int)

    return stock_actualizado
