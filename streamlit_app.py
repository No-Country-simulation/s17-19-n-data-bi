import streamlit as st
import pandas as pd
import google.generativeai as genai
import streamlit.components.v1 as components
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
    st.subheader("Inicie Sesión")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    # Aquí puedes cargar usuarios de un archivo CSV o una base de datos
    if st.button("INGRESAR"):
        if autenticar_usuario(username, password):
            st.success("USTED INGRESÓ EXITOSAMENTE!")
            return True
        else:
            st.error("Usuario o contraseña incorrectos! Reintentar nuevamente.")
            return False

# Autenticación básica usando un archivo CSV
def autenticar_usuario(username, password):
    try:
        # Limpiar las entradas del usuario (eliminar espacios y convertir todo a minúsculas)
        username = username.strip().lower()
        password = password.strip()

        # Cargar los usuarios desde el archivo CSV y limpiar las columnas
        users_df = pd.read_csv("users.csv")
        users_df['username'] = users_df['username'].astype(str).str.strip().str.lower()
        users_df['password'] = users_df['password'].astype(str).str.strip()

        # Verificar si hay coincidencia
        user_row = users_df[(users_df['username'] == username) & (users_df['password'] == password)]
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
    if 'selected_button' not in st.session_state:
        st.session_state['selected_button'] = None  

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
    
    st.sidebar.image('streamlit_app/Pi.png', use_column_width=True)
    st.sidebar.title("Bienvenid@! Selecciona el insight:")
    
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

    if st.sidebar.button('ANÁLISIS INTEGRAL'):
        st.session_state['selected_button'] = 'ANÁLISIS INTEGRAL'

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
        
            powerbi_urls = [
            "https://app.powerbi.com/view?r=eyJrIjoiMWM5MTYzNTgtZjcxZC00ZTJhLTg4YjctOTZjYmVmYWM3MTIzIiwidCI6ImE0NDRiYjgyLTYzYjYtNDkxMi05Nzg1LTE5ZDhmODRiNzY3OCIsImMiOjR9"
            ]

            titles = [
            "Análisis y Reportes"       
            ]

            for title, url in zip(titles, powerbi_urls):
                st.markdown(f"### {title}")
                components.html(
                    f"""
                    <iframe width="900" height="700" src="{url}" frameborder="0" allowFullScreen="true"></iframe>
                    """,
                    height=700,
                )
                st.markdown("---")


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
            country = st.text_input('Ingrese el país')
            region = st.text_input('Ingrese la región')
            therapeutic_group = st.text_input('Ingrese el grupo terapéutico')
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
    
    
    elif st.session_state['selected_button'] == 'ANÁLISIS INTEGRAL':
        st.title("Distribuciones y Análisis")
        
        def mostrar_dashboard(title, url, width=900, height=700):
            """
            Función para mostrar un dashboard de Power BI embebido en Streamlit.
            
            :param title: Título del dashboard
            :param url: URL del dashboard
            :param width: Ancho del iframe
            :param height: Alto del iframe
            """
            st.markdown(f"### {title}")
            components.html(
                f"""
                <iframe width="{width}" height="{height}" src="{url}" frameborder="0" allowFullScreen="true"></iframe>
                """,
                height=height,
            )
            st.markdown("---")  # Separador
        
        # Listas de títulos y URLs
        titles = ["Análisis de Productos, Riesgo y Cobertura", "Análisis por Sucursales, de Consumo de Producto y Tendencias"]
        powerbi_urls = [
            "https://app.powerbi.com/view?r=eyJrIjoiMzM3M2U1MWUtYTM3OS00YjY5LTljMzYtZjNhMjUzNWQ3Mzk5IiwidCI6ImE0NDRiYjgyLTYzYjYtNDkxMi05Nzg1LTE5ZDhmODRiNzY3OCIsImMiOjR9", 
            "https://app.powerbi.com/view?r=eyJrIjoiOWJiNDYyOGMtNzE2OS00NjhkLTgxMTUtMjc0NmY4M2RhYzRkIiwidCI6ImE0NDRiYjgyLTYzYjYtNDkxMi05Nzg1LTE5ZDhmODRiNzY3OCIsImMiOjR9"
        ]
        
        # Mostrar los dashboards con mayor altura para que ambos se rendericen correctamente
        for title, url in zip(titles, powerbi_urls):
            mostrar_dashboard(title, url, width=900, height=700)
        

# Función para suscribir al cliente al newsletter
def suscribir_a_newsletter(email):
    # Verificar si ya existe el archivo CSV, si no, crearlo
    archivo = "newsletter_subscribers.csv"
    if os.path.exists(archivo):
        # Leer el archivo existente
        df = pd.read_csv(archivo)
    else:
        # Crear un nuevo DataFrame si no existe el archivo
        df = pd.DataFrame(columns=["email"])

    # Verificar si el correo ya está suscrito
    if email in df['email'].values:
        st.warning("Este correo ya está suscrito al newsletter.")
    else:
        # Añadir el correo al DataFrame y guardar en el archivo CSV
        nuevo_suscriptor = pd.DataFrame({"email": [email]})
        df = pd.concat([df, nuevo_suscriptor], ignore_index=True)
        df.to_csv(archivo, index=False)
        st.success("Te has suscrito exitosamente al newsletter de ofertas!")


# Lógica para clientes
def mostrar_lógica_cliente():
    if 'selected_button' not in st.session_state:
        st.session_state['selected_button'] = None    

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

    st.sidebar.image('streamlit_app/Pi.png', use_column_width=True)
    st.sidebar.title("Bienvenid@! Selecciona tu consulta:")

    if st.sidebar.button('SUSCRIBIRSE AL NEWSLETTER'):
        st.session_state['selected_button'] = 'SUSCRIBIRSE AL NEWSLETTER'
    
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


    elif st.session_state.get('selected_button') == 'VERIFICAR COBERTURA':
        st.title("Alcance de Cobertura según Perfil Terapéutico")

        df = pd.read_parquet("data/Productos.parquet")
        # Extraer los perfiles terapéuticos únicos del DataFrame
        perfiles_terapeuticos = df['perfil_terapeutico'].unique()
        nombres_medicamentos = df['descriprod_agrp2'].unique()

        # Input del cliente: Perfil Terapéutico utilizando un selectbox
        perfil_terapeutico = st.selectbox("Seleccione el perfil terapéutico:", perfiles_terapeuticos)

        # Filtrar los medicamentos según el perfil terapéutico seleccionado
        medicamentos_filtrados = df[df["perfil_terapeutico"] == perfil_terapeutico]["descriprod_agrp2"].unique()

        # Input del cliente: Nombre del Medicamento utilizando un selectbox
        medicamento = st.selectbox("Seleccione el nombre del medicamento:", medicamentos_filtrados)

        # Botón de búsqueda
        if st.button("Consultar"):
            # Filtrar según el perfil terapéutico y el nombre del medicamento
            filtro = df[(df["perfil_terapeutico"] == perfil_terapeutico) & 
                        (df["descriprod_agrp2"].str.contains(medicamento, case=False, na=False))]
            
            # Verificar si el medicamento fue encontrado
            if not filtro.empty:
                # Mostrar los resultados de la consulta
                clasificacion = filtro["cobertura_contrato"].values[0]
                generico = filtro["lineaproducto"].values[0]  # Indica si es genérico
                
                # Resultado al cliente
                st.write(f"**Clasificación PBS (con cobertura) o NO PBS (sin cobertura). Resultado**: {clasificacion}")
                
                if generico == "GENERICOS":
                    st.write("Este medicamento tiene una variante genérica.")
                else:
                    st.write("Este medicamento **NO** tiene una variante genérica.")
            else:
                st.write("No hay información disponible. Acérquese a la Sucursal más cercana para hacer su consulta.")

      
    elif st.session_state['selected_button'] == 'CUIDÁ TU SALUD, CONSULTÁ !':
        st.title("Campañas de Información y Prevención Vigentes")
        with st.form(key='afinidad_form'):
            prompt = st.text_input("Ingrese su consulta")
            submit_button = st.form_submit_button("Enviar Consulta")
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

    
    # Lógica de suscripción al newsletter
    elif st.session_state['selected_button'] == 'SUSCRIBIRSE AL NEWSLETTER':
        st.title("Suscríbete a nuestro Newsletter de Ofertas y Campañas de Salud")
        email_cliente = st.text_input("Ingrese su correo electrónico")
        if st.button("SUSCRIBIRSE"):
            if email_cliente:
                suscribir_a_newsletter(email_cliente)
            else:
                st.warning("Por favor, ingrese un correo electrónico válido.")


def main():
    st.image('Pi.png', width=550)
    st.title("   Bienvenid@! Seleccione su perfil:")

    option = st.selectbox("Elija una opción", ("Seleccione", "SOY FARMACÉUTICA", "SOY CLIENTE"))

    if option == "SOY FARMACÉUTICA":
        # Verificamos si ya está autenticado en la sesión
        if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
            if mostrar_login():
                st.session_state['logged_in'] = True  # Almacenar que el usuario inició sesión
                mostrar_lógica_farmacéutica()
        else:
            mostrar_lógica_farmacéutica()  # Ya está autenticado, mostramos la lógica directamente

    elif option == "SOY CLIENTE":
        mostrar_lógica_cliente()  # Muestra directamente la lógica para clientes

if __name__ == "__main__":
    main()

