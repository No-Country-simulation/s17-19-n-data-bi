# Código en proceso aún...

import streamlit as st
import torch
import os
from models.inference import predict, load_model  # Asegúrate de que la ruta a inference.py esté bien configurada
from models.marketing_model import configure_gemini_api as configure_marketing_gemini_api, get_promotion_suggestions
from models.afinidad_model import configure_gemini_api as configure_afinidad_gemini_api, get_related_products

# Configurar la API de Gemini para marketing y afinidad de productos
configure_marketing_gemini_api()
configure_afinidad_gemini_api()

# Definir la arquitectura del modelo para Previsión Generativa
class PrevisionModel(torch.nn.Module):
    def __init__(self):
        super(PrevisionModel, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)  # Ajusta las dimensiones según tu entrada real
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Cargar el logo
st.sidebar.image('streamlit_app/Pi.png', use_column_width=True)

# Título en el menú lateral
st.sidebar.title("Bienvenid@! Selecciona el insight:")

# Aplicar estilo CSS para botones verdes
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

# Lógica para cada botón seleccionado
if gestion_stocks:
    # Lógica para Gestión de Stocks
    model_name = "stock"
    model = load_model(model_name)
    if model:
        sucursal_id = st.text_input("Ingrese el ID de la sucursal")
        skuagr_2 = st.text_input("Ingrese el SKU del producto")

        if st.button('Verificar Stock'):
            try:
                input_data = torch.tensor([float(sucursal_id), float(skuagr_2)])  # Ajusta los datos según sea necesario
                result = predict(model, input_data)
                st.write(f'Resultado de la predicción: {result}')
            except ValueError:
                st.error("Por favor ingrese valores numéricos válidos.")

elif prevision_consumo:
    st.sidebar.subheader("Selecciona el método de previsión:")
    prevision_basada_datos = st.sidebar.button('PREVISIÓN BASADA EN DATOS')
    prevision_generativa = st.sidebar.button('PREVISIÓN CON GenAI')

    if prevision_basada_datos:
        st.title('Previsión Basada en Datos')
        # Aquí integras las visualizaciones de Power BI
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
                    input_data = torch.tensor([float(country), float(region), float(context), float(season)])  # Ajusta según corresponda
                    result = predict(model, input_data)
                    st.write(f'Resultado de la previsión: {result}')
                except ValueError:
                    st.error("Por favor ingrese valores numéricos válidos.")
                # Lógica adicional con la API de Gemini
                input_prompt = (
                    f"En el país {country}, región {region}, con un contexto epidemiológico de {context}, "
                    f"dado que estamos en la {season}, ¿qué previsiones de consumo se deberían tomar en cuenta?"
                )
                response = genai.generate_text(input_prompt)
                if response and hasattr(response, 'text'):
                    st.write("Previsión GenAI sugerida:")
                    st.write(response.text)
                else:
                    st.warning("No se pudo generar una respuesta adecuada.")

elif marketing_intelligence:
    st.title('Marketing Intelligence')

    # Entradas de usuario
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
    st.title('Afinidad de Productos')

    # Entrada de usuario
    main_product = st.text_input('Ingrese el nombre del producto o categoría principal:')

    if st.button('Buscar Productos Relacionados'):
        if not main_product:
            st.warning("Por favor, ingrese un producto o categoría.")
        else:
            related_products = get_related_products(main_product)
            if related_products:
                st.subheader(f'Productos relacionados con {main_product}')
                for i, product in enumerate(related_products, 1):
                    st.write(f"{i}. {product}")
            else:
                st.warning("No se encontraron productos relacionados.")
