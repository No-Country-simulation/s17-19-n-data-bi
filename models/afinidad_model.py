import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Configurar la API generativa de Gemini
def configure_gemini_api():
    load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=gemini_api_key)

def get_related_products(main_product, num_suggestions=5):
    if not main_product:
        return ["Por favor, proporciona un producto o categoría para obtener sugerencias."]

    input_prompt = (
        f"Si alguien está buscando {main_product}, ¿qué otros productos relacionados "
        f"son comúnmente comprados o utilizados junto con él? Sugiere {num_suggestions} productos relacionados."
    )

    try:
        response = genai.generate_text(input_prompt)
        if response and hasattr(response, 'text'):
            related_products = response.text.split('\n')
            if len(related_products) >= num_suggestions:
                return related_products[:num_suggestions]  # Devolver solo las primeras `num_suggestions`
            else:
                return related_products
        else:
            return ["No se pudo generar una respuesta adecuada."]
    except Exception as e:
        return [f"Error al generar productos relacionados: {e}"]

# Streamlit app
configure_gemini_api()

st.title("Sugerencias de Productos Relacionados")

# Usar un formulario para evitar recargas al cambiar de campo
with st.form(key='related_products_form'):
    main_product = st.text_input("Producto o Categoría Principal")
    num_suggestions = st.number_input("Número de Sugerencias", min_value=1, max_value=10, value=5)
    
    # Botón para enviar el formulario
    submit_button = st.form_submit_button(label='Obtener Productos Relacionados')

# Solo generar sugerencias si se ha enviado el formulario
if submit_button:
    suggestions = get_related_products(main_product, num_suggestions)
    st.write("Productos Relacionados Sugeridos:")
    for i, suggestion in enumerate(suggestions, start=1):
        st.write(f"{i}. {suggestion}")
