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
    prompt = f"Genera 10 sugerencias de promociones específicas para el grupo terapéutico de {therapeutic_group} en {country}, {region}. Primero mostrarás el título de referencia, luego cada promoción debe estar enumerada, debe incluir un título, un ítem de descripción y un ítem de valor agregado. Debe respetar una estructura clara."

    try:
        response = model.generate_content([prompt])

        if response and hasattr(response, 'text'):
            suggestions = response.text.splitlines()

            # Inicializar una lista para almacenar las promociones formateadas
            formatted_suggestions = []
            current_promotion = []

            for line in suggestions:
                stripped_line = line.strip()
                
                # Comienza una nueva promoción cuando se encuentra un número al principio de la línea
                if stripped_line and stripped_line[0].isdigit():
                    if current_promotion:
                        formatted_suggestions.append(" ".join(current_promotion))
                    current_promotion = [stripped_line]
                else:
                    current_promotion.append(stripped_line)

            # Añadir la última promoción a la lista
            if current_promotion:
                formatted_suggestions.append(" ".join(current_promotion))

            # Devolver sólo las 10 primeras sugerencias
            return formatted_suggestions[:10]

        else:
            return ["No se pudieron generar sugerencias."]
    
    except Exception as e:
        raise RuntimeError(f"Error al generar las sugerencias: {e}")

