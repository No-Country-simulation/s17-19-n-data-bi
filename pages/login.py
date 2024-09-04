import pandas as pd
import streamlit as st

# Función para cargar el CSV con usuarios y contraseñas
def cargar_usuarios():
    return pd.read_csv("users.csv")  # Ruta del archivo CSV

# Función para verificar las credenciales
def autenticar_usuario(username, password):
    usuarios = cargar_usuarios()
    
    # Verificamos si el usuario existe y si la contraseña es correcta
    if not usuarios[(usuarios['username'] == username) & (usuarios['password'] == password)].empty:
        return True
    else:
        return False

# Lógica de autenticación
def mostrar_login():
    st.subheader("Login para Farmacéuticas")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Login"):
        if autenticar_usuario(username, password):
            st.success("Login exitoso")
            return True
        else:
            st.error("Usuario o contraseña incorrectos")
            return False
