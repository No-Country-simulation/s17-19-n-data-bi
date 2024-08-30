import streamlit as st
import streamlit.session_state as session
import pandas as pd

# Función para cargar usuarios desde un archivo CSV
def load_users():
    users_df = pd.read_csv('users.csv')
    return dict(zip(users_df.username, users_df.password))

# Función de autenticación
def authenticate(username, password, users):
    return username in users and users[username] == password

def main():
    st.title("Página de Login")

    users = load_users()

    if 'authenticated' not in session:
        session.authenticated = False

    if not session.authenticated:
        # Formulario de login
        st.subheader("Inicia sesión")

        username = st.text_input("Nombre de usuario")
        password = st.text_input("Contraseña", type="password")
        if st.button("Iniciar sesión"):
            if authenticate(username, password, users):
                session.authenticated = True
                session.username = username
                st.success("¡Has iniciado sesión exitosamente!")
                st.experimental_rerun()
            else:
                st.error("Nombre de usuario o contraseña incorrectos")
    else:
        st.success("Ya estás autenticado.")
        st.write("Ve a la página de inicio para explorar la aplicación.")

if __name__ == "__main__":
    main()
