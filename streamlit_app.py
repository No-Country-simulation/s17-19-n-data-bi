# Código en proceso aún...

import streamlit as st
import boto3
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv


# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Acceder a la clave API desde el archivo .env
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Configurar la API generativa de Gemini
genai.configure(api_key=gemini_api_key)

# Función para generar sugerencias de promociones usando la API generativa
def get_promotion_suggestions(country, region, therapeutic_group):
    input_prompt = (
        f"En el país {country}, región {region}, y para el grupo terapéutico {therapeutic_group}, "
        "sugiere 5 promociones de marketing farmacéutico efectivas que podrían implementarse. "
        "Las promociones deben estar alineadas con las tendencias actuales de mercado y ser aplicables en el contexto local. "
        "Hacer una oferta de implementación de un precio sugerido en la moneda local del país seleccionado. "
    )

    try:
        response = genai.generate_text(input_prompt)
        if response and hasattr(response, 'text'):
            suggestions = response.text.split('\n')
            return suggestions[:5]  # Devolver solo las primeras 5 sugerencias
        else:
            return ["No se pudo generar una respuesta adecuada."]
    except Exception as e:
        st.error(f"Error al generar sugerencias: {e}")
        return ["Error generando las sugerencias."]

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

# Aplicar el estilo a los botones
st.markdown(button_style, unsafe_allow_html=True)

# Crear botones en el menú lateral
gestion_stocks = st.sidebar.button('GESTIÓN DE STOCKS')
prevision_consumo = st.sidebar.button('PREVISIÓN DE CONSUMO')
marketing_intelligence = st.sidebar.button('MARKETING INTELLIGENCE')
afinidad_productos = st.sidebar.button('AFINIDAD DE PRODUCTOS')
demanda_pcb = st.sidebar.button('DEMANDA DE PCB & NO PCB')

# Configura el cliente de SageMaker
sagemaker_client = boto3.client('sagemaker-runtime', region_name='us-west-2')

# Nombre del endpoint que obtuviste después de desplegar el modelo en SageMaker
endpoint_name = 'my-endpoint-name'  # Reemplaza con el nombre real de tu endpoint

# Contenido para GESTIÓN DE STOCKS
if gestion_stocks:
    st.title('Verificación de Stock en Sucursales')

    # Input de usuario para seleccionar la sucursal y el producto
    sucursal_id = st.text_input("Ingrese el ID de la sucursal")
    skuagr_2 = st.text_input("Ingrese el SKU del producto")

    # Cuando el usuario hace clic en el botón, enviar los datos al endpoint de SageMaker
    if st.button('Verificar Stock'):
        # Crear el payload que se enviará al endpoint
        payload = json.dumps({
            'sucursal_id': sucursal_id,
            'skuagr_2': skuagr_2
        })

        # Llamar al endpoint de SageMaker
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )

        # Leer la respuesta
        result = json.loads(response['Body'].read().decode())
        st.write(f'Resultado de la predicción: {result}')

#elif prevision_consumo:
#    st.title('Previsión de Consumo')
#    st.write('Aquí puedes implementar la lógica para la previsión de consumo.')

# Configura el estado de la sesión
if 'country' not in st.session_state:
    st.session_state['country'] = ''
if 'region' not in st.session_state:
    st.session_state['region'] = ''
if 'therapeutic_group' not in st.session_state:
    st.session_state['therapeutic_group'] = ''

# Contenido para cada botón seleccionado
if marketing_intelligence:
    st.title('Marketing Intelligence')

    # Entradas de usuario con manejo de estado
    st.session_state['country'] = st.text_input('Ingrese el país:', st.session_state['country'])
    st.session_state['region'] = st.text_input('Ingrese la región / estado / provincia:', st.session_state['region'])
    st.session_state['therapeutic_group'] = st.text_input('Ingrese el grupo terapéutico:', st.session_state['therapeutic_group'])

    # Botón para obtener sugerencias de promociones
    if st.button('Obtener Sugerencias de Promociones'):
        if not st.session_state['country'] or not st.session_state['region'] or not st.session_state['therapeutic_group']:
            st.warning("Por favor, ingrese todos los campos requeridos.")
        else:
            suggestions = get_promotion_suggestions(
                st.session_state['country'], 
                st.session_state['region'], 
                st.session_state['therapeutic_group']
            )
            
            # Mostrar las sugerencias
            if suggestions:
                st.subheader('Sugerencias de Promociones')
                for i, suggestion in enumerate(suggestions, 1):
                    st.write(f"Promoción {i}: {suggestion}")
            else:
                st.warning("No se encontraron sugerencias para las opciones seleccionadas.")

#elif afinidad_productos:
#    st.title('Afinidad de Productos')
#   st.write('Aquí puedes implementar la lógica para afinidad de productos.')

#elif demanda_pcb:
#    st.title('Demanda de PCB & NO PCB')
#    st.write('Aquí puedes implementar la lógica para demanda de PCB & NO PCB.')
