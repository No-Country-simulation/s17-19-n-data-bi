import streamlit as st
import torch
import os
import google.generativeai as genai
from models.inference import load_model, predict
from models.stock_logic import stock_verification
from models.marketing_model import get_promotion_suggestions, configure_gemini_api
from models.afinidad_model import get_related_products
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

# Obtener la API Key de Gemini
gemini_api_key = os.getenv('GEMINI_API_KEY')

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

# Cargar el modelo una vez al iniciar la aplicación
try:
    stock_model = load_model('stock')  # Cargar el modelo de stock
except FileNotFoundError as e:
    st.error(f"Error al cargar el modelo de stock: {e}")
    stock_model = None  # Establecer el modelo como None si no se carga correctamente
except RuntimeError as e:
    st.error(f"Error al inicializar el modelo de stock: {e}")
    stock_model = None

# Crear botones en el menú lateral
gestion_stocks = st.sidebar.button('GESTIÓN DE STOCKS')
prevision_consumo = st.sidebar.button('PREVISIÓN DE CONSUMO')
marketing_intelligence = st.sidebar.button('MARKETING INTELLIGENCE')
afinidad_productos = st.sidebar.button('AFINIDAD DE PRODUCTOS')

# Gestionar la lógica de cada botón
if gestion_stocks:
    if stock_model is not None:
        stock_verification(stock_model)  # Pasar el modelo cargado a la función
    else:
        st.warning("El modelo de stock no está disponible. No se puede verificar el stock.")

if prevision_consumo:
    st.title("Selecciona el Método de Previsión")

    with st.form(key='prevision_form'):
        col1, col2 = st.columns(2)
        metodo = None
        with col1:
            prevision_basada_datos = st.form_submit_button('PREVISIÓN BASADA EN DATOS')
            if prevision_basada_datos:
                metodo = 'datos'
        with col2:
            prevision_generativa = st.form_submit_button('PREVISIÓN CON GenAI')
            if prevision_generativa:
                metodo = 'generativa'

        # Almacenamos el método seleccionado en una variable
        st.session_state['metodo_seleccionado'] = metodo

    # Mostrar el contenido basado en el método seleccionado
    if st.session_state.get('metodo_seleccionado') == 'datos':
        st.subheader('Previsión Basada en Datos')
        st.markdown("[Visualización de Power BI](URL_DE_TU_POWER_BI)")

    elif st.session_state.get('metodo_seleccionado') == 'generativa':
        st.subheader('Previsión Con GenAI')
        st.write("Escribe una previsión de consumo que se desea saber según estación del año, contexto epidemiológico, país y región de ese país.")
        
        user_prompt = st.text_area("PROMPT:", height=250)
        
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
    with st.form(key='marketing_form'):
        country = st.text_input('Ingrese el país:')
        region = st.text_input('Ingrese la región / estado / provincia:')
        therapeutic_group = st.text_input('Ingrese el grupo terapéutico:')
        
        # Botón para enviar el formulario
        submit_button = st.form_submit_button('Obtener Sugerencias de Promociones')

    if submit_button:
        suggestions = get_promotion_suggestions(country, region, therapeutic_group)
        if suggestions:
            st.subheader('Sugerencias de Promociones')
            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"Promoción {i}: {suggestion}")
        else:
            st.warning("No se encontraron sugerencias para las opciones seleccionadas.")

if afinidad_productos:
    st.title("Posibles Demandas de Productos Relacionados")
    with st.form(key='afinidad_form'):
        prompt = st.text_input("Ingrese un producto o categoría para ver productos relacionados")
        
        # Botón para enviar el formulario
        submit_button = st.form_submit_button("Generar Afinidad de Productos")

    if submit_button:
        if prompt:
            suggestions = get_related_products(prompt)
            if suggestions:
                st.subheader("Productos relacionados sugeridos:")
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"Producto relacionado {i}: {suggestion}")
        else:
            st.warning("Por favor, ingrese un producto o categoría.")
