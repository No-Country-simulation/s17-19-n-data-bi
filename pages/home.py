import streamlit as st
import streamlit.session_state as session

def main():
    st.title("BIENVENID@")

    if 'authenticated' in session and session.authenticated:
        st.write(f"Bienvenid@ {session['username']}, esta es la página principal.")
        # Aquí puedes agregar más contenido, como gráficos, tablas, etc.
    else:
        st.error("Debes iniciar sesión para acceder a esta página.")
        st.write("Por favor, ve a la página de Login utilizando la barra lateral.")

if __name__ == "__main__":
    main()

