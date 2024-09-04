import streamlit as st
import pandas as pd
import google.generativeai as genai
from models.inference import load_model, predict
from models.stock_logic import stock_verification
from models.marketing_model import get_promotion_suggestions
from models.afinidad_model import get_affinity_recommendations
from models.stock_logic_u import stock_verification as stock_verification_u
from models.care_model import get_care_recommendations

# Configurar Gemini API
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Cargar modelo de stock
try:
    stock_model = load_model('stock')
except FileNotFoundError as e:
    st.error(f"Error al cargar el modelo de stock: {e}")
    stock_model = None
except RuntimeError as e:
    st.error(f"Error al inicializar el modelo de stock: {e}")
    stock_model = None

# Función para manejar el login
def mostrar_login():
    st.subheader("Login para Farmacéuticas")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    # Aquí puedes cargar usuarios de un archivo CSV o una base de datos
    if st.button("Login"):
        if autenticar_usuario(username, password):
            st.success("Login exitoso")
            return True
        else:
            st.error("Usuario o contraseña incorrectos")
            return False

# Autenticación básica usando un archivo CSV
def autenticar_usuario(username, password):
    try:
        users_df = pd.read_csv("users.csv")  # Asegúrate de tener un archivo CSV con usuarios y contraseñas
        user_row = users_df[(users_df["username"] == username) & (users_df["password"] == password)]
        return not user_row.empty
    except Exception as e:
        st.error(f"Error al cargar el archivo de usuarios: {e}")
        return False

# Función para generar respuestas con GenAI
def generate_gemini_response(input_prompt):
    try:
        response = model.generate_content([input_prompt])
        if response and hasattr(response, 'text'):
            return response.text
        elif response and hasattr(response, 'generated_text'):
            return response.generated_text
        else:
            return "No se pudo generar una respuesta adecuada."
    except IndexError as e:
        st.error(f"Error al generar la respuesta: {e}")
        return "Error de índice al generar la respuesta."
    except Exception as e:
        st.error(f"Error al generar la respuesta: {e}")
        return "Error generando la respuesta."

# Lógica para farmacéuticos
def mostrar_lógica_farmacéutica():
    # Cargar el logo en la barra lateral
    st.sidebar.image('streamlit_app/Pi.png', use_column_width=True)
    st.sidebar.title("Bienvenid@! Selecciona el insight:")

    # Aplicar estilo CSS a los botones
    button_style = """
        <style>
        .stButton > button {
            width: 100%;
            height: 50px;
            background-color: #96ffae;
            color: dark blue;
            font-size: 16px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    # Crear botones en el menú lateral y manejar el estado
    if st.sidebar.button('GESTIÓN DE STOCKS'):
        st.session_state['selected_button'] = 'GESTIÓN DE STOCKS'

    if st.sidebar.button('PREVISIÓN DE CONSUMO'):
        st.session_state['selected_button'] = 'PREVISIÓN DE CONSUMO'
        st.session_state['show_prevision_generativa'] = False
        st.session_state['generated_prevision'] = None

    if st.sidebar.button('MARKETING INTELLIGENCE'):
        st.session_state['selected_button'] = 'MARKETING INTELLIGENCE'

    if st.sidebar.button('AFINIDAD DE PRODUCTOS'):
        st.session_state['selected_button'] = 'AFINIDAD DE PRODUCTOS'

    if st.sidebar.button('PRODUCTOS CON COBERTURA O SIN COBERTURA'):
        st.session_state['selected_button'] = 'PRODUCTOS CON COBERTURA O SIN COBERTURA'

    if st.session_state['selected_button'] == 'GESTIÓN DE STOCKS':
        st.title("Verificación de Stock Disponible en Sucursales")
        if stock_model is not None:
            stock_verification()
        else:
            st.warning("El modelo de stock no está disponible.")

    elif st.session_state['selected_button'] == 'PREVISIÓN DE CONSUMO':
        st.title("Selecciona tu Consulta de Interés")
        col1, col2 = st.columns(2)
        with col1:
            prevision_basada_datos = st.button('PREVISIÓN BASADA EN DATOS')
        with col2:
            prevision_generativa = st.button('PREVISIÓN CON GenAI')

        if prevision_basada_datos:
            st.subheader('Previsión Basada en Datos')
            st.markdown("[Visualización de Power BI](URL_DE_TU_POWER_BI)")

        if prevision_generativa:
            st.session_state['show_prevision_generativa'] = True

        if st.session_state['show_prevision_generativa']:
            st.subheader('Previsión Con GenAI')
            st.write("Escribe una previsión de consumo según estación, contexto epidemiológico, etc.")
            user_prompt = st.text_area("PROMPT:", height=250)
            if st.button("Generar Previsión"):
                if user_prompt.strip() == "":
                    st.warning("Por favor, ingresa una consulta.")
                else:
                    st.write("Procesando...")
                    try:
                        response = generate_gemini_response(user_prompt)
                        if response:
                            st.session_state['generated_prevision'] = response
                        else:
                            st.warning("No se pudo generar una previsión adecuada.")
                    except Exception as e:
                        st.error(f"Error al generar la previsión: {e}")
            if st.session_state['generated_prevision']:
                st.write("### Previsión generada con GenAI:")
                st.success(st.session_state['generated_prevision'])

    elif st.session_state['selected_button'] == 'MARKETING INTELLIGENCE':
        st.title('Sistema de Recomendación de Precios y Combos')
        with st.form(key='marketing_form'):
            country = st.text_input('Ingrese el país:')
            region = st.text_input('Ingrese la región:')
            therapeutic_group = st.text_input('Ingrese el grupo terapéutico:')
            submit_button = st.form_submit_button('Obtener Sugerencias de Promociones')
        if submit_button:
            if not country or not region or not therapeutic_group:
                st.warning("Por favor, complete todos los campos.")
            else:
                try:
                    suggestions = get_promotion_suggestions(country, region, therapeutic_group)
                    if suggestions:
                        st.subheader('Sugerencias de Promociones')
                        st.write("\n".join(suggestions))
                    else:
                        st.warning("No se encontraron sugerencias.")
                except Exception as e:
                    st.error(f"Error al obtener sugerencias: {e}")

    elif st.session_state['selected_button'] == 'AFINIDAD DE PRODUCTOS':
        st.title("Posibles Demandas de Productos Relacionados")
        with st.form(key='afinidad_form'):
            prompt = st.text_input("Ingrese un producto o categoría")
            submit_button = st.form_submit_button("Generar Afinidad de Productos")
        if submit_button:
            if prompt:
                try:
                    suggestions = get_affinity_recommendations(prompt)
                    if suggestions:
                        st.subheader("Productos relacionados sugeridos:")
                        st.write("\n".join(suggestions))
                    else:
                        st.warning("No se encontraron productos relacionados.")
                except Exception as e:
                    st.error(f"Error al obtener productos relacionados: {e}")
            else:
                st.warning("Por favor, ingrese un producto o categoría.")

    elif st.session_state['selected_button'] == 'PRODUCTOS CON COBERTURA O SIN COBERTURA':
        st.title("Alcance de Cobertura según Perfil Terapéutico")
        st.subheader('Visualizaciones de Power BI')
        st.markdown("[Visualización de Power BI PRODUCTOS CON COBERTURA](URL_DE_TU_POWER_BI_1)")
        st.markdown("[Visualización de Power BI PRODUCTOS SIN COBERTURA](URL_DE_TU_POWER_BI_2)")
        st.markdown("[Visualización de Power BI PRODUCTOS GENÉRICOS](URL_DE_TU_POWER_BI_3)")

# Lógica para clientes
def mostrar_lógica_cliente():
    st.sidebar.image('streamlit_app/Pi.png', use_column_width=True)
    st.sidebar.title("Bienvenid@! Selecciona el insight:")
    
    if st.sidebar.button('HAY STOCK ?'):
        st.session_state['selected_button'] = 'HAY STOCK ?'

    if st.sidebar.button('VERIFICAR COBERTURA'):
        st.session_state['selected_button'] = 'VERIFICAR COBERTURA'

    if st.sidebar.button('CUIDÁ TU SALUD, CONSULTÁ !'):
        st.session_state['selected_button'] = 'CUIDÁ TU SALUD, CONSULTÁ !'

    if st.session_state['selected_button'] == 'HAY STOCK ?':
        st.title("Verificación de Stock Disponible en Sucursales")
        if stock_model is not None:
            stock_verification_u()
        else:
            st.warning("La consulta de stock no está disponible.")

    elif st.session_state['selected_button'] == 'VERIFICAR COBERTURA':
        st.title("Alcance de Cobertura según Perfil Terapéutico")
        st.subheader('Visualizaciones de Power BI')
        st.markdown("[Visualización de Power BI PRODUCTOS CON COBERTURA](URL_DE_TU_POWER_BI_1)")
        st.markdown("[Visualización de Power BI PRODUCTOS SIN COBERTURA](URL_DE_TU_POWER_BI_2)")

    elif st.session_state['selected_button'] == 'CUIDÁ TU SALUD, CONSULTÁ !':
        st.title("Campañas de Información y Prevención Vigentes")
        with st.form(key='afinidad_form'):
            prompt = st.text_input("Ingrese su consulta:")
            submit_button = st.form_submit_button("ENVIAR CONSULTA")
        if submit_button:
            if prompt:
                try:
                    suggestions = get_care_recommendations(prompt)
                    if suggestions:
                        st.subheader("Respuesta:")
                        st.write("\n".join(suggestions))
                    else:
                        st.warning("No se encontraron respuestas.")
                except Exception as e:
                    st.error(f"Error al obtener respuesta: {e}")
            else:
                st.warning("Por favor, ingrese su consulta.")

def main():
    st.image('Pi.png', width=400)
    st.title("   Bienvenid@! Seleccione su perfil:")

    option = st.selectbox("Elija una opción", ("Seleccione", "SOY FARMACÉUTICA", "SOY CLIENTE"))

    if option == "SOY FARMACÉUTICA":
        if mostrar_login():
            mostrar_lógica_farmacéutica()  # Si el login es exitoso, muestra la lógica para farmacéuticos

    elif option == "SOY CLIENTE":
        mostrar_lógica_cliente()  # Muestra directamente la lógica para clientes

if __name__ == "__main__":
    main()
