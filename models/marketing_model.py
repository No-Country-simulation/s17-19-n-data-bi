import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("API key for Gemini not found. Make sure it's set as an environment variable.")

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
    prompt = (
        f"Genera una lista de al menos 10 sugerencias específicas de promociones para productos farmacéuticos o de belleza "
        f"relacionados con {therapeutic_group} en {country}, {region}. Cada sugerencia debe ser clara y enfocada en "
        f"promociones prácticas para consumidores. Evita descripciones genéricas y concéntrate en ofertas específicas."
        f"Separa las sugerencias como una lista enumerada del 1 al 10, renglón por renglón."
    )

    try:
        response = genai.generate_text(prompt)
        
        if response and hasattr(response, 'text'):
            suggestions = response.text.strip().splitlines()

            # Filtrar líneas vacías y devolver hasta 10 sugerencias
            return [s for s in suggestions if s][:10]

        else:
            return ["No se pudieron generar sugerencias."]
    
    except Exception as e:
        raise RuntimeError(f"Error al generar las sugerencias: {e}")
