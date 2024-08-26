import google.generativeai as genai
import os
from dotenv import load_dotenv

# Configurar la API generativa de Gemini
def configure_gemini_api():
    load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=gemini_api_key)

def get_promotion_suggestions(country, region, therapeutic_group, num_suggestions=5):
    # Validar que los parámetros no estén vacíos
    if not country or not region or not therapeutic_group:
        return ["Por favor, proporciona el país, la región y el grupo terapéutico para obtener sugerencias."]

    input_prompt = (
        f"En el país {country}, región {region}, y para el grupo terapéutico {therapeutic_group}, "
        f"sugiere {num_suggestions} promociones de marketing farmacéutico efectivas que podrían implementarse. "
        "Las promociones deben estar alineadas con las tendencias actuales de mercado y ser aplicables en el contexto local. "
        "Hacer una oferta de implementación de un precio sugerido en la moneda local del país seleccionado."
    )

    try:
        response = genai.generate_text(input_prompt)
        if response and hasattr(response, 'text'):
            suggestions = response.text.split('\n')
            if len(suggestions) >= num_suggestions:
                return suggestions[:num_suggestions]  # Devolver solo las primeras `num_suggestions`
            else:
                return suggestions
        else:
            return ["No se pudo generar una respuesta adecuada."]

    except Exception as e:
        return [f"Error al generar sugerencias: {e}"]
