# -*- coding: utf-8 -*-
"""train_stock_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_X1kcvtCX7PxFi9LwQzqdQfO2PsCZ3v5
"""

import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar los datasets
sucursales = pd.read_parquet('data/Sucursales.parquet')
productos = pd.read_parquet('data/Productos.parquet')
data = pd.read_parquet('data/Data.parquet')
usuarios = pd.read_parquet('data/Usuarios.parquet')

# ENTRENAMIENTO PARA STOCK_MODEL.PTH

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


# Definir X e y
X = stock_actualizado.drop(['stock_inicial', 'cantidad_dispensada', 'stock_disponible', 'hay_stock'], axis=1)
y = stock_actualizado['hay_stock']

# Convertir las características categóricas a variables dummy
X = pd.get_dummies(X, drop_first=True)

# Dividir en conjuntos de entrenamiento y prueba de manera estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar los datos como antes y continuar el resto del proceso
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train)
X_test_np = scaler.transform(X_test)

# Convertir a tensores para PyTorch
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

from imblearn.over_sampling import SMOTE

# Aplicar SMOTE solo al conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Escalar los datos
X_train_np = scaler.fit_transform(X_train_smote)
X_test_np = scaler.transform(X_test)

# Convertir a tensores
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_smote.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Definir el modelo
class StockModel(nn.Module):
    def __init__(self):
        super(StockModel, self).__init__()
        self.fc1 = nn.Linear(X_train_tensor.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Crear una instancia del modelo
model = StockModel()

# Definir la función de pérdida y el optimizador
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Variables para guardar el mejor modelo
best_val_loss = float('inf')  # Iniciar con una pérdida muy alta

# Entrenar el modelo por 30 épocas
for epoch in range(30):
    # Entrenamiento
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    # Evaluar en conjunto de validación
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
    
    # Guardar el mejor modelo según la pérdida de validación
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_stock_model.pth')

    # Imprimir la pérdida de entrenamiento y validación
    print(f'Epoch [{epoch+1}/30], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Crear la carpeta 'models' si no existe
os.makedirs('models', exist_ok=True)

# Guardar el modelo final
torch.save(model.state_dict(), 'models/stock_model.pth')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Evaluar el modelo en el conjunto de prueba
model.eval()  # Cambiar el modelo a modo de evaluación
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = (y_pred > 0.5).float()  # Convertir las probabilidades en clases (0 o 1)

# Convertir los tensores de PyTorch a numpy para usar con sklearn
y_pred_np = y_pred_class.numpy()
y_test_np = y_test_tensor.numpy()

# Calcular las métricas
accuracy = accuracy_score(y_test_np, y_pred_np)
precision = precision_score(y_test_np, y_pred_np, zero_division=0)
recall = recall_score(y_test_np, y_pred_np, zero_division=0)
f1 = f1_score(y_test_np, y_pred_np, zero_division=0)

# Comprobar si hay ambas clases en y_test antes de calcular el ROC-AUC
if len(np.unique(y_test_np)) > 1:
    roc_auc = roc_auc_score(y_test_np, y_pred)
    print(f"ROC-AUC: {roc_auc:.4f}")
else:
    print("Solo hay una clase presente en y_test, no se puede calcular el ROC-AUC.")

# Imprimir las métricas
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
