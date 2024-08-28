import google.generativeai as genai
import streamlit as st

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
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

def generate_gemini_response(input_prompt):
    try:
        response = model.generate_content(input_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Error generando la respuesta."

def get_promotion_suggestions(country, region, therapeutic_group):
    prompt = (
        f"Genera una lista de al menos 10 sugerencias específicas de promociones para productos farmacéuticos o de belleza "
        f"relacionados con {therapeutic_group} en {country}, {region}. Cada sugerencia debe ser clara y enfocada en "
        f"promociones prácticas para consumidores. Evita descripciones genéricas y concéntrate en ofertas específicas. "
        f"Separa las sugerencias como una lista enumerada del 1 al 10, renglón por renglón."
    )
    
    try:
        suggestions = generate_gemini_response(prompt)
        
        # Procesamos la respuesta para obtener una lista de sugerencias
        suggestion_list = suggestions.strip().splitlines()
        # Filtramos líneas vacías y limitamos a 10 sugerencias
        suggestion_list = [s for s in suggestion_list if s][:10]
        
        if suggestion_list:
            return suggestion_list
        else:
            return ["No se pudieron generar sugerencias."]
    
    except Exception as e:
        st.error(f"Error al generar las sugerencias: {e}")
        return ["Error al generar las sugerencias."]


# import streamlit as st
# import google.generativeai as genai
# import os

# GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# genai.configure(api_key=GEMINI_API_KEY)

# generation_config = {
#     "temperature": 0.4,
#     "top_p": 1,
#     "top_k": 32,
#     "max_output_tokens": 4096,
# }

# safety_settings = [
#     {
#         "category": "HARM_CATEGORY_HARASSMENT",
#         "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#     },
#     {
#         "category": "HARM_CATEGORY_HATE_SPEECH",
#         "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#     },
#     {
#         "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#         "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#     },
#     {
#         "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#         "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#     }
# ]

# model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
#                               generation_config=generation_config,
#                               safety_settings=safety_settings)

# def get_promotion_suggestions(country, region, therapeutic_group):
#     prompt = (
#         f"Genera una lista de al menos 10 sugerencias específicas de promociones para productos farmacéuticos o de belleza "
#         f"relacionados con {therapeutic_group} en {country}, {region}. Cada sugerencia debe ser clara y enfocada en "
#         f"promociones prácticas para consumidores. Evita descripciones genéricas y concéntrate en ofertas específicas."
#         f"Separa las sugerencias como una lista enumerada del 1 al 10, renglón por renglón."
#     )

#     try:
#         response = genai.generate_text(prompt=prompt)
        
#         if response and hasattr(response, 'text'):
#             suggestions = response.text.strip().splitlines()

#             # Filtrar líneas vacías y devolver hasta 10 sugerencias
#             return [s for s in suggestions if s][:10]

#         else:
#             return ["No se pudieron generar sugerencias."]
    
#     except Exception as e:
#         raise RuntimeError(f"Error al generar las sugerencias: {e}")
