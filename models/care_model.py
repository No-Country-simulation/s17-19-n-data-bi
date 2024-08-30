import streamlit as st
import google.generativeai as genai

# Configuración de la API de Gemini
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
if GEMINI_API_KEY is None:
    raise Exception("API key for Gemini not found. Make sure it's set in the secrets.toml file.")

genai.configure(api_key=GEMINI_API_KEY)

# Configuración del modelo
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

def generate_gemini_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Error generando la respuesta."

def get_care_recommendations(prompt, language="es"):
    prompt_text = (
        f"Genera una lista de posibles consejos para el cuidado de la salud, con enfoque preventivo, informativo, o de recomendación"
        f"según la petición del usuario: '{prompt}'. Debe estar en {language} y cada enfoque debe ser específico y útil para "
        f"el usuario final. Asegúrate de que las recomendaciones estén numeradas y sean claras y prácticas."
    )
    try:
        response = generate_gemini_response(prompt_text)
        if response and response != "Error generando la respuesta.":
            suggestions = response.strip().splitlines()
            # Filtrar líneas vacías y devolver hasta 10 sugerencias
            return [s for s in suggestions if s][:11]
        else:
            return ["No se pudo generar su consulta."]
    
    except Exception as e:
        st.error(f"Error al generar la consulta: {e}")
        return ["Error al generar la consulta."]
