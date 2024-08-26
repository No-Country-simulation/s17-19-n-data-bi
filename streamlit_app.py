# Código en proceso aún...

import streamlit as st
import torch
import os
from models.inference import predict, load_model
from models.marketing_model import configure_gemini_api, get_promotion_suggestions
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

# Load the logo
st.sidebar.image('streamlit_app/Pi.png', use_column_width=True)

# Sidebar title
st.sidebar.title("Bienvenid@! Selecciona el insight:")

# Apply CSS style to buttons
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

# Ensure input data persistence using session_state
if 'country' not in st.session_state:
    st.session_state['country'] = ''
if 'region' not in st.session_state:
    st.session_state['region'] = ''
if 'therapeutic_group' not in st.session_state:
    st.session_state['therapeutic_group'] = ''
if 'context' not in st.session_state:
    st.session_state['context'] = ''
if 'season' not in st.session_state:
    st.session_state['season'] = ''

# Create buttons in the sidebar
gestion_stocks = st.sidebar.button('GESTIÓN DE STOCKS')
prevision_consumo = st.sidebar.button('PREVISIÓN DE CONSUMO')
marketing_intelligence = st.sidebar.button('MARKETING INTELLIGENCE')
afinidad_productos = st.sidebar.button('AFINIDAD DE PRODUCTOS')
demanda_pcb = st.sidebar.button('DEMANDA DE PCB & NO PCB')

# Handling the model selection and UI display
if gestion_stocks:
    st.title("Verificación de Stock en Sucursales")
    model_name = "stock"
    model = load_model(model_name)

    if model:
        sucursal_id = st.text_input("Ingrese el ID de la sucursal", key="sucursal_id")
        skuagr_2 = st.text_input("Ingrese el SKU del producto", key="skuagr_2")

        if st.button('Verificar Stock'):
            try:
                input_data = torch.tensor([float(sucursal_id), float(skuagr_2)])
                result = predict(model, input_data)
                st.write(f'Resultado de la predicción: {result}')
            except ValueError:
                st.error("Por favor ingrese valores numéricos válidos.")
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")

elif prevision_consumo:
    st.title("Previsión de Consumo")
    st.session_state['prevision_method'] = st.radio(
        "Selecciona el método de previsión:",
        ('PREVISIÓN BASADA EN DATOS', 'PREVISIÓN CON GenAI')
    )

    if st.session_state['prevision_method'] == 'PREVISIÓN BASADA EN DATOS':
        st.subheader('Previsión Basada en Datos')
        st.markdown("[Visualización de Power BI](URL_DE_TU_POWER_BI)")

    elif st.session_state['prevision_method'] == 'PREVISIÓN CON GenAI':
        st.subheader('Previsión Con GenAI')
        model_name = "prevision"
        model = load_model(model_name)

        if model:
            st.session_state['country'] = st.text_input("Ingrese el país", st.session_state['country'])
            st.session_state['region'] = st.text_input("Ingrese la región / estado / provincia", st.session_state['region'])
            st.session_state['context'] = st.text_input("Contexto epidemiológico", st.session_state['context'])
            st.session_state['season'] = st.text_input("Época del año", st.session_state['season'])

            if st.button('Generar Previsión'):
                try:
                    input_data = torch.tensor([
                        float(st.session_state['country']),
                        float(st.session_state['region']),
                        float(st.session_state['context']),
                        float(st.session_state['season'])
                    ])
                    result = predict(model, input_data)
                    st.write(f'Resultado de la previsión: {result}')
                except ValueError:
                    st.error("Por favor ingrese valores numéricos válidos.")
                except Exception as e:
                    st.error(f"Error al realizar la previsión: {e}")

                # Generative AI part
                input_prompt = (
                    f"En el país {st.session_state['country']}, región {st.session_state['region']}, "
                    f"con un contexto epidemiológico de {st.session_state['context']}, "
                    f"dado que estamos en la {st.session_state['season']}, ¿qué previsiones de consumo se deberían tomar en cuenta?"
                )
                try:
                    response = genai.generate_text(input_prompt)
                    if response and hasattr(response, 'text'):
                        st.write("Previsión GenAI sugerida:")
                        st.write(response.text)
                    else:
                        st.warning("No se pudo generar una respuesta adecuada.")
                except Exception as e:
                    st.error(f"Error al generar previsión con GenAI: {e}")

elif marketing_intelligence:
    st.title('Marketing Intelligence')
    configure_gemini_api()

    st.session_state['country'] = st.text_input('Ingrese el país:', st.session_state['country'])
    st.session_state['region'] = st.text_input('Ingrese la región / estado / provincia:', st.session_state['region'])
    st.session_state['therapeutic_group'] = st.text_input('Ingrese el grupo terapéutico:', st.session_state['therapeutic_group'])

    if st.button('Obtener Sugerencias de Promociones'):
        if not st.session_state['country'] or not st.session_state['region'] or not st.session_state['therapeutic_group']:
            st.warning("Por favor, ingrese todos los campos requeridos.")
        else:
            suggestions = get_promotion_suggestions(
                st.session_state['country'], 
                st.session_state['region'], 
                st.session_state['therapeutic_group']
            )
            if suggestions:
                st.subheader('Sugerencias de Promociones')
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"Promoción {i}: {suggestion}")
            else:
                st.warning("No se encontraron sugerencias para las opciones seleccionadas.")

elif afinidad_productos:
    st.title("Afinidad de Productos")
    from models.afinidad_model import get_related_products

    prompt = st.text_input("Ingrese un producto o categoría para ver productos relacionados", key="afinidad_prompt")
    if st.button("Generar afinidad de productos"):
        if prompt:
            suggestions = get_related_products(prompt)
            if suggestions:
                st.subheader("Productos relacionados sugeridos:")
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"Producto relacionado {i}: {suggestion}")
        else:
            st.warning("Por favor, ingrese un producto o categoría.")

                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"Producto relacionado {i}: {suggestion}")
        else:
            st.warning("Por favor, ingrese un producto o categoría.")

