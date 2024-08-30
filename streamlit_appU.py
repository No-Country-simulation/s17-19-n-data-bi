import streamlit as st
import torch
import os
import google.generativeai as genai
from models.inference import load_model, predict
from models.stock_logic_u import stock_verification
from models.care_model import get_care_recommendations

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
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

model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

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
if st.sidebar.button('HAY STOCK ?'):
    st.session_state['selected_button'] = 'HAY STOCK ?'

if st.sidebar.button('VERIFICAR COBERTURA'):
    st.session_state['selected_button'] = 'VERIFICAR COBERTURA'

if st.sidebar.button('CUIDÁ TU SALUD, CONSULTÁ !'):
    st.session_state['selected_button'] = 'AFINIDAD DE PRODUCTOS'

# Gestionar la lógica basada en el botón presionado
if st.session_state['selected_button'] == 'HAY STOCK ?':
    st.title("Verificación de Stock Disponible en Sucursales")
    if stock_model is not None:
        stock_verification()
    else:
        st.warning("La consulta de stock no está disponible. Por favor, acérquese a su sucursal más cercana.")

def generate_gemini_response(input_prompt):
    try:
        # Generar contenido basado en el prompt del usuario
        response = model.generate_content([input_prompt])
        
        # Acceder al texto generado directamente desde el objeto de respuesta
        if response and hasattr(response, 'text'):
            return response.text
        elif response and hasattr(response, 'generated_text'):
            return response.generated_text
        else:
            return "No se pudo generar una respuesta adecuada."
    except IndexError as e:
        st.error(f"IndexError generating response: {e}")
        return "Error de índice al generar la respuesta."
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Error generando la respuesta."

if st.session_state['selected_button'] == 'VERIFICAR COBERTURA':
    st.title("Alcance de Cobertura según Perfil Terapéutico")
    st.subheader('Visualizaciones de Power BI')
    # Aquí puedes agregar múltiples enlaces a diferentes visualizaciones de Power BI
    st.markdown("[Visualización de Power BI PRODUCTOS CON COBERTURA](URL_DE_TU_POWER_BI_1)")
    st.markdown("[Visualización de Power BI PRODUCTOS SIN COBERTURA](URL_DE_TU_POWER_BI_2)")
    st.markdown("[Visualización de Power BI PRODUCTOS GENÉRICOS](URL_DE_TU_POWER_BI_3)")

elif st.session_state['selected_button'] == 'CUIDÁ TU SALUD, CONSULTÁ !':
    st.title("Campañas de Información y Prevención Vigentes")
    with st.form(key='afinidad_form'):
        prompt = st.text_input("Ingrese su consulta:")
        
        # Botón para enviar el formulario
        submit_button = st.form_submit_button("ENVIAR CONSULTA")

    if submit_button:
        if prompt:
            try:
                # Llamada a la función para obtener afinidades de productos
                suggestions = get_care_recommendations(prompt)
                
                if suggestions:
                    st.subheader("Respuesta:")
                    st.write("\n".join(suggestions))
                else:
                    st.warning("No se encontraron respuestas a su consulta.")
            except Exception as e:
                st.error(f"Se produjo un error al obtener respuesta: {e}")
        else:
            st.warning("Por favor, ingrese nuevamente su consulta.")


