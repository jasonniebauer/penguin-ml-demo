import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.title('Penguin Species Classification Demo')

        st.info(
            "The purpose of this application is to provide a simple experience of the process of creating an ML model and releasing a web application that uses that model."
        )

        st.page_link('streamlit_app.py', label='Predict', icon=':material/smart_toy:')
        st.page_link(
            'pages/training-data.py',
            label='Training Data',
            icon=':material/database:',
        )