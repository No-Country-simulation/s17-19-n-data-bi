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
    prompt = f"Genera 10 sugerencias de promociones específicas para el grupo terapéutico de {therapeutic_group} en {country}, {region}. Cada promoción debe incluir un título, un ítem de descripción y un ítem de valor agregado. Debe respetar una estructura clara y concisa."

    try:
        response = model.generate_content([prompt])

        if response and hasattr(response, 'text'):
            suggestions = response.text.splitlines()

            # Inicializar una lista para almacenar las promociones formateadas
            formatted_suggestions = []
            current_promotion = ""
            in_title_block = False

            for line in suggestions:
                stripped_line = line.strip()

                # Detectar si estamos en el bloque del título general
                if stripped_line.startswith("##"):
                    formatted_suggestions.append(stripped_line)
                elif stripped_line.startswith("1."):
                    # Fin del bloque de título general, inicio de las promociones
                    formatted_suggestions.append("\nGrupo Terapéutico de " + therapeutic_group + f" - {region}, {country}:\n")
                    current_promotion = f"Promoción {stripped_line[0]}: {stripped_line[3:].strip()}"
                elif stripped_line and stripped_line[0].isdigit() and "." in stripped_line[1:3]:
                    # Comienza una nueva promoción
                    if current_promotion:
                        formatted_suggestions.append(current_promotion.strip())
                    current_promotion = f"Promoción {stripped_line[:2].strip()}: {stripped_line[3:].strip()}"
                elif stripped_line.startswith("-") or stripped_line.startswith("*"):
                    # Continuación de la descripción o valor agregado
                    current_promotion += f"\n{stripped_line}"
                else:
                    # Continuación de la línea anterior
                    current_promotion += f" {stripped_line}"

            # Añadir la última promoción a la lista
            if current_promotion:
                formatted_suggestions.append(current_promotion.strip())

            # Devolver sólo las 10 primeras sugerencias
            return formatted_suggestions[:10]

        else:
            return ["No se pudieron generar sugerencias."]
    
    except Exception as e:
        raise RuntimeError(f"Error al generar las sugerencias: {e}")
