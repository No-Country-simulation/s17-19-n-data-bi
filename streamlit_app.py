import streamlit as st
import torch
import os
import google.generativeai as genai
from models.inference import load_model, predict
from models.stock_logic import stock_verification
from models.marketing_model import get_promotion_suggestions, configure_gemini_api
from models.afinidad_model import get_related_products
from dotenv import load_dotenv

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

if GEMINI_API_KEY is None:
    raise Exception("API key for Gemini not found. Make sure it's set in the config.toml file.")

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Cargar el modelo de stock una vez al iniciar la aplicación
try:
    stock_model = load_model('stock')
except FileNotFoundError as e:
    st.error(f"Error al cargar el modelo de stock: {e}")
    stock_model = None
except RuntimeError as e:
    st.error(f"Error al inicializar el modelo de stock: {e}")
    stock_model = None

# Cargar el logo en la barra lateral
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
        margin-bottom: 10px;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Inicializar el estado de la sesión para manejar los botones
if 'selected_button' not in st.session_state:
    st.session_state['selected_button'] = None

# Crear botones en el menú lateral y manejar el estado
if st.sidebar.button('GESTIÓN DE STOCKS'):
    st.session_state['selected_button'] = 'GESTIÓN DE STOCKS'

if st.sidebar.button('PREVISIÓN DE CONSUMO'):
    st.session_state['selected_button'] = 'PREVISIÓN DE CONSUMO'
    st.session_state['show_prevision_generativa'] = False
    st.session_state['generated_prevision'] = None

if st.sidebar.button('MARKETING INTELLIGENCE'):
    st.session_state['selected_button'] = 'MARKETING INTELLIGENCE'

if st.sidebar.button('AFINIDAD DE PRODUCTOS'):
    st.session_state['selected_button'] = 'AFINIDAD DE PRODUCTOS'

if st.sidebar.button('PRODUCTOS CON COBERTURA O SIN COBERTURA'):
    st.session_state['selected_button'] = 'PRODUCTOS CON COBERTURA O SIN COBERTURA'

# Gestionar la lógica basada en el botón presionado
if st.session_state['selected_button'] == 'GESTIÓN DE STOCKS':
    st.title("Verificación de Stock Disponible en Sucursales")
    if stock_model is not None:
        stock_verification()
    else:
        st.warning("El modelo de stock no está disponible. No se puede verificar el stock.")

def fake_generate_text(prompt):
    return f"Resultado de la previsión para: {prompt}"

# Estado inicial
if 'selected_button' not in st.session_state:
    st.session_state['selected_button'] = None
if 'show_prevision_generativa' not in st.session_state:
    st.session_state['show_prevision_generativa'] = False
if 'generated_prevision' not in st.session_state:
    st.session_state['generated_prevision'] = None

if st.session_state['selected_button'] == 'PREVISIÓN DE CONSUMO':
    st.title("Selecciona tu Consulta de Interés")
    col1, col2 = st.columns(2)

    with col1:
        prevision_basada_datos = st.button('PREVISIÓN BASADA EN DATOS')
    with col2:
        prevision_generativa = st.button('PREVISIÓN CON GenAI')

    if prevision_basada_datos:
        st.subheader('Previsión Basada en Datos')
        st.markdown("[Visualización de Power BI](URL_DE_TU_POWER_BI)")

    if prevision_generativa:
        st.session_state['show_prevision_generativa'] = True

    if st.session_state['show_prevision_generativa']:
        st.subheader('Previsión Con GenAI')
        st.write("Escribe una previsión de consumo que se desea saber según estación del año, contexto epidemiológico, país y región de ese país.")
        
        user_prompt = st.text_area("PROMPT:", height=250)
        
        if st.button("Generar Previsión"):
            if user_prompt.strip() == "":
                st.warning("Por favor, ingresa una consulta en el área de texto antes de generar la previsión.")
            else:
                st.write("Procesando tu solicitud...")
    
                try:
                    # Simulación de la generación de previsión
                    response = fake_generate_text(prompt=user_prompt)
                    
                    if response:
                        st.session_state['generated_prevision'] = response
                    else:
                        st.warning("No se pudo generar una previsión adecuada. Inténtalo de nuevo.")
    
                except Exception as e:
                    st.error(f"Error al generar la previsión: {e}")

        # Mostrar la previsión generada en un lugar visible
        if st.session_state['generated_prevision']:
            st.write("### Previsión generada con GenAI:")
            st.success(st.session_state['generated_prevision'])

elif st.session_state['selected_button'] == 'MARKETING INTELLIGENCE':
    st.title('Sistema de Recomendación de Precios y Combos')
    with st.form(key='marketing_form'):
        country = st.text_input('Ingrese el país:')
        region = st.text_input('Ingrese la región / estado / provincia:')
        therapeutic_group = st.text_input('Ingrese el grupo terapéutico:')
        
        # Botón para enviar el formulario
        submit_button = st.form_submit_button('Obtener Sugerencias de Promociones')

    if submit_button:
        if not country or not region or not therapeutic_group:
            st.warning("Por favor, complete todos los campos antes de obtener sugerencias.")
        else:
            try:
                suggestions = get_promotion_suggestions(country, region, therapeutic_group)
                if suggestions:
                    st.subheader('Sugerencias de Promociones')
                    for i, suggestion in enumerate(suggestions, 1):
                        st.write(f"Promoción {i}: {suggestion}")
                else:
                    st.warning("No se encontraron sugerencias para las opciones seleccionadas.")
            except Exception as e:
                st.error(f"Se produjo un error al obtener las sugerencias: {e}")

elif st.session_state['selected_button'] == 'AFINIDAD DE PRODUCTOS':
    st.title("Posibles Demandas de Productos Relacionados")
    with st.form(key='afinidad_form'):
        prompt = st.text_input("Ingrese un contexto, un producto o una categoría para ver combinaciones relacionadas")
        
        # Botón para enviar el formulario
        submit_button = st.form_submit_button("Generar Afinidad de Productos")

    if submit_button:
        if prompt:
            try:
                suggestions = get_related_products(prompt)
                if suggestions:
                    st.subheader("Productos relacionados sugeridos:")
                    for i, suggestion in enumerate(suggestions, 1):
                        st.write(f"Producto relacionado {i}: {suggestion}")
                else:
                    st.warning("No se encontraron productos relacionados.")
            except Exception as e:
                st.error(f"Se produjo un error al obtener los productos relacionados: {e}")
        else:
            st.warning("Por favor, ingrese un producto o categoría.")

elif st.session_state['selected_button'] == 'PRODUCTOS CON COBERTURA O SIN COBERTURA':
    st.title("Productos con Cobertura o Sin Cobertura")
    st.subheader('Visualizaciones de Power BI')
    # Aquí puedes agregar múltiples enlaces a diferentes visualizaciones de Power BI
    st.markdown("[Visualización de Power BI PRODUCTOS CON COBERTURA](URL_DE_TU_POWER_BI_1)")
    st.markdown("[Visualización de Power BI PRODUCTOS SIN COBERTURA](URL_DE_TU_POWER_BI_2)")
    st.markdown("[Visualización de Power BI PRODUCTOS GENÉRICOS](URL_DE_TU_POWER_BI_3)")

