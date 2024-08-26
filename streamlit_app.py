# Código en proceso aún...

import streamlit as st
import torch
import os
import google.generativeai as genai
from models.inference import load_model, predict
from models.marketing_model import get_promotion_suggestions, configure_gemini_api
from models.afinidad_model import get_related_products
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

def configure_gemini_api():
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        st.error("No se encontró la API Key de Gemini. Por favor, verifica tu archivo .env.")
        return
    genai.configure(api_key=gemini_api_key)

configure_gemini_api()

# Cargar el logo
st.sidebar.image('streamlit_app/Pi.png', use_column_width=True)

# Título en el menú lateral
st.sidebar.title("Bienvenid@! Selecciona el insight:")

# Aplicar estilo CSS a los botones
button_style = """
    <style>
    .stButton > button {
        width: 100%;
        height: 50px;
        background-color: #96ffae;
        color: dark blue;
        font-size: 16px;
        border-radius: 10px;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Crear botones en el menú lateral
gestion_stocks = st.sidebar.button('GESTIÓN DE STOCKS')
prevision_consumo = st.sidebar.button('PREVISIÓN DE CONSUMO')
marketing_intelligence = st.sidebar.button('MARKETING INTELLIGENCE')
afinidad_productos = st.sidebar.button('AFINIDAD DE PRODUCTOS')

# Gestionar la lógica de cada botón
if gestion_stocks:
    st.title("Verificación de Stock en Sucursales")
    model = load_model("stock")

    if model:
        sucursal_id = st.text_input("Ingrese el ID de la sucursal")
        skuagr_2 = st.text_input("Ingrese el SKU del producto")

        if st.button('Verificar Stock'):
            try:
                input_data = torch.tensor([float(sucursal_id), float(skuagr_2)])
                result = predict(model, input_data)
                st.write(f'Resultado de la predicción: {result}')
            except ValueError:
                st.error("Por favor ingrese valores numéricos válidos.")
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")

if prevision_consumo:
    st.title("Selecciona el Método de Previsión")
    
    col1, col2 = st.columns(2)
    with col1:
        prevision_basada_datos = st.button('PREVISIÓN BASADA EN DATOS', key="btn_prevision_basada_datos")
    with col2:
        prevision_generativa = st.button('PREVISIÓN CON GenAI', key="btn_prevision_generativa")

    if prevision_basada_datos:
        st.subheader('Previsión Basada en Datos')
        st.markdown("[Visualización de Power BI](URL_DE_TU_POWER_BI)")

    if prevision_generativa:
        st.subheader('Previsión Con GenAI')
        st.write("Aquí se manejaría la lógica para la previsión generativa con GenAI.")
        
        st.write("Escribe una previsión de consumo que se desea saber según estación del año, contexto epidemiológico, país y región de ese país.")
        
        user_prompt = st.text_area("Escribe aquí la previsión de consumo que deseas saber:", height=250)
        
        if st.button("Generar Previsión"):
            if user_prompt.strip() == "":
                st.warning("Por favor, ingresa una consulta en el área de texto antes de generar la previsión.")
            else:
                st.write("Procesando tu solicitud...")
    
                try:
                    # Usar la función de generación de texto de GenAI
                    response = genai.generate_text(prompt=user_prompt)
                    
                    if response and hasattr(response, 'text'):
                        st.write("Previsión generada con GenAI:")
                        st.write(response.text)
                    else:
                        st.warning("No se pudo generar una previsión adecuada. Inténtalo de nuevo.")
    
                except Exception as e:
                    st.error(f"Error al generar la previsión: {e}")

if marketing_intelligence:
    st.title('Sistema de Recomendación de Precios y Combos')
    country = st.text_input('Ingrese el país:')
    region = st.text_input('Ingrese la región / estado / provincia:')
    therapeutic_group = st.text_input('Ingrese el grupo terapéutico:')

    if st.button('Obtener Sugerencias de Promociones'):
        suggestions = get_promotion_suggestions(country, region, therapeutic_group)
        if suggestions:
            st.subheader('Sugerencias de Promociones')
            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"Promoción {i}: {suggestion}")
        else:
            st.warning("No se encontraron sugerencias para las opciones seleccionadas.")

if afinidad_productos:
    st.title("Posibles Demandas de Productos Relacionados")
    prompt = st.text_input("Ingrese un producto o categoría para ver productos relacionados")

    if st.button("Generar Afinidad de Productos"):
        if prompt:
            suggestions = get_related_products(prompt)
            if suggestions:
                st.subheader("Productos relacionados sugeridos:")
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"Producto relacionado {i}: {suggestion}")
        else:
            st.warning("Por favor, ingrese un producto o categoría.")
