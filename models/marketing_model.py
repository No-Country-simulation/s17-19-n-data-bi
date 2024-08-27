import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

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

def get_promotion_suggestions(country, region, therapeutic_group):
    prompt = f"Genera posibles promociones de productos farmacéuticos o de belleza, según los datos que el usuario ha ingresado, grupo terapéutico de {therapeutic_group} en {country}, {region}."

    try:
        response = model.generate_content([prompt])

        if response and hasattr(response, 'text'):
            # Dividir la respuesta en líneas, cada línea sería una sugerencia
            suggestions = response.text.strip().splitlines()
            
            # Devolver sólo las 10 primeras sugerencias
            return suggestions[:10] if len(suggestions) > 10 else suggestions

        else:
            return ["No se pudieron generar sugerencias."]
    
    except Exception as e:
        raise RuntimeError(f"Error al generar las sugerencias: {e}")
        
