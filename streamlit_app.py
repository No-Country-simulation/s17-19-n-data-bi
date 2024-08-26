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

elif prevision_consumo:
    st.sidebar.subheader("Selecciona el método de previsión:")
    prevision_basada_datos = st.sidebar.button('PREVISIÓN BASADA EN DATOS')
    prevision_generativa = st.sidebar.button('PREVISIÓN CON GenAI')

    if prevision_basada_datos:
        st.title('Previsión Basada en Datos')
        st.markdown("[Visualización de Power BI](URL_DE_TU_POWER_BI)")

    elif prevision_generativa:
        st.title('Previsión Con GenAI')
        model_name = "prevision"
        model = load_model(model_name)

        if model:
            country = st.text_input("Ingrese el país")
            region = st.text_input("Ingrese la región / estado / provincia")
            context = st.text_input("Contexto epidemiológico")
            season = st.text_input("Época del año")

            if st.button('Generar Previsión'):
                try:
                    input_data = torch.tensor([float(country), float(region), float(context), float(season)])
                    result = predict(model, input_data)
                    st.write(f'Resultado de la previsión: {result}')
                except ValueError:
                    st.error("Por favor ingrese valores numéricos válidos.")
                except Exception as e:
                    st.error(f"Error al realizar la previsión: {e}")

                # Generative AI part
                input_prompt = (
                    f"En el país {country}, región {region}, con un contexto epidemiológico de {context}, "
                    f"dado que estamos en la {season}, ¿qué previsiones de consumo se deberían tomar en cuenta?"
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

    country = st.text_input('Ingrese el país:')
    region = st.text_input('Ingrese la región / estado / provincia:')
    therapeutic_group = st.text_input('Ingrese el grupo terapéutico:')

    if st.button('Obtener Sugerencias de Promociones'):
        if not country or not region or not therapeutic_group:
            st.warning("Por favor, ingrese todos los campos requeridos.")
        else:
            suggestions = get_promotion_suggestions(country, region, therapeutic_group)
            if suggestions:
                st.subheader('Sugerencias de Promociones')
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"Promoción {i}: {suggestion}")
            else:
                st.warning("No se encontraron sugerencias para las opciones seleccionadas.")

elif afinidad_productos:
    st.title("Afinidad de Productos")
    from models.afinidad_model import get_related_products

    prompt = st.text_input("Ingrese un producto o categoría para ver productos relacionados")
    if st.button("Generar afinidad de productos"):
        if prompt:
            suggestions = get_related_products(prompt)
            if suggestions:
                st.subheader("Productos relacionados sugeridos:")
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"Producto relacionado {i}: {suggestion}")
        else:
            st.warning("Por favor, ingrese un producto o categoría.")

