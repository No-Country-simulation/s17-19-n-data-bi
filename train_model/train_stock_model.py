# -*- coding: utf-8 -*-
"""train_stock_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_X1kcvtCX7PxFi9LwQzqdQfO2PsCZ3v5
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Cargar los datasets
sucursales = pd.read_parquet('data/Sucursales.parquet')
productos = pd.read_parquet('data/Productos.parquet')
data = pd.read_parquet('data/Data.parquet')
usuarios = pd.read_parquet('data/Usuarios.parquet')

# ENTRENAMIENTO PARA STOCK_MODEL.PTH

# Crear la tabla de stock inicial y calcular el stock disponible
stock_inicial = pd.merge(sucursales[['id_sucursal']], productos[['skuagr_2']], how='cross')
stock_inicial['stock_inicial'] = 300  # Asignar 100 unidades como stock inicial

# Sumar las transacciones para cada combinación de sucursal y producto
transacciones_agrupadas = data.groupby(['id_sucursal', 'skuagr_2']).agg({'cantidad_dispensada': 'sum'}).reset_index()
stock_actualizado = pd.merge(stock_inicial, transacciones_agrupadas, on=['id_sucursal', 'skuagr_2'], how='left')
stock_actualizado['cantidad_dispensada'] = stock_actualizado['cantidad_dispensada'].fillna(0)
stock_actualizado['stock_disponible'] = stock_actualizado['stock_inicial'] - stock_actualizado['cantidad_dispensada']

# Crear la variable objetivo
stock_actualizado['hay_stock'] = stock_actualizado['stock_disponible'] > 0
stock_actualizado['hay_stock'] = stock_actualizado['hay_stock'].astype(int)

# Revisar los datos procesados
stock_actualizado.head()

# Definir X e y
X = stock_actualizado.drop(['stock_inicial', 'cantidad_dispensada', 'stock_disponible', 'hay_stock'], axis=1)
y = stock_actualizado['hay_stock']

# Convertir las características categóricas a variables dummy
X = pd.get_dummies(X, drop_first=True)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train)
X_test_np = scaler.transform(X_test)

# Convertir los datos a tensores para PyTorch
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
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

# Entrenar el modelo
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

# Crear la carpeta 'models' si no existe
os.makedirs('models', exist_ok=True)

# Guardar los pesos del modelo
torch.save(model.state_dict(), 'models/stock_model.pth')
