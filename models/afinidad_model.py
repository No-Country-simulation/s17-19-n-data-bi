import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Configura la API de Gemini
GEMINI_API_KEY = "your_gemini_api_key"  # Asegúrate de que esta clave esté bien configurada

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

def get_affinity_recommendations(product, country, region):
    prompt = (
        f"Genera una lista de recomendaciones de productos que tengan afinidad con {product} en {country}, {region}. "
        f"Incluye sugerencias específicas que sean relevantes para los consumidores en esa área."
    )

    try:
        response = model.generate_content([prompt])

        if response and hasattr(response, 'text'):
            suggestions = response.text.strip().splitlines()

            # Filtrar líneas vacías y devolver sólo recomendaciones útiles
            return [s for s in suggestions if s]

        else:
            return ["No se pudieron generar recomendaciones."]
    
    except Exception as e:
        raise RuntimeError(f"Error al generar las recomendaciones: {e}")
