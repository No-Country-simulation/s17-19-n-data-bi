import streamlit as st
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

# Streamlit app
configure_gemini_api()

st.title("Sugerencias de Promociones de Marketing Farmacéutico")

with st.form(key='marketing_form'):
    country = st.text_input("País")
    region = st.text_input("Región")
    therapeutic_group = st.text_input("Grupo Terapéutico")
    num_suggestions = st.number_input("Número de Sugerencias", min_value=1, max_value=10, value=5)
    
    # Botón para enviar el formulario
    submit_button = st.form_submit_button(label='Obtener Sugerencias')

# Solo generar sugerencias si se ha enviado el formulario
if submit_button:
    suggestions = get_promotion_suggestions(country, region, therapeutic_group, num_suggestions)
    st.write("Sugerencias de Promociones:")
    for i, suggestion in enumerate(suggestions, start=1):
        st.write(f"{i}. {suggestion}")

        else:
            return ["No se pudo generar una respuesta adecuada."]

    except Exception as e:
        return [f"Error al generar sugerencias: {e}"]
