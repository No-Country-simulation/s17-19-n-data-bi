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
