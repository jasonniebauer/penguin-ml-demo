import streamlit as st
import numpy as np
import pandas as pd
from prediction import predict, predict_probability


st.set_page_config(
    page_title="Penguin Species Classification Demo",
    page_icon=":penguin:",
)

def render_sidebar():
    with st.sidebar:
        st.title('Penguin Species Classification Demo')

        st.info(
            "The purpose of this application is to provide a simple experience of the process of creating an ML model and releasing a web application that uses that model."
        )

        st.page_link('streamlit_app.py', label='Predict', icon=':material/smart_toy:')
        st.page_link(
            'pages/training_data.py',
            label='Training Data',
            icon=':material/database:',
        )

with st.container(border=True):
    st.markdown('##### Input Features')

    with st.spinner():

        col1, col2 = st.columns(2)

        with col1:
            sex = st.selectbox('Sex', ('male', 'female'))
        with col2:
            island = st.selectbox(
                'Island', (
                    'Biscoe',
                    'Dream',
                    'Torgersen',
                )
            )

        bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)

        # Create Dataframe for the input features
        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex,
        }
        input_df = pd.DataFrame(data, index=[0])

        # Data encoding for category variables
        encode = ['island', 'sex']
        input_encoded_df = pd.get_dummies(input_df, prefix=encode)

        # Ensure all dummy variables used during model training are present in this order
        expected_columns = [
            'bill_length_mm',
            'bill_depth_mm',
            'flipper_length_mm',
            'body_mass_g',
            'island_Biscoe',
            'island_Dream',
            'island_Torgersen',
            'sex_female',
            'sex_male',
        ]

        # Add missing category variables as columns with 0 value
        for col in expected_columns:
            if col not in input_encoded_df.columns:
                input_encoded_df[col] = False

        # Reorder df_penguins in line with expected_columns
        input_encoded_df = input_encoded_df[expected_columns]

        # Execute prediction
        prediction = predict(input_encoded_df)
        prediction_probability = predict_probability(input_encoded_df)
        prediction_probability = [n * 100 for n in prediction_probability]

        # Display prediction result
        st.write('##### 🐧Species prediction results')
        penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
        st.success(str(penguins_species[prediction][0]))

        # Display prediction probability
        df_prediction_proba = pd.DataFrame(prediction_probability)
        df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
        df_prediction_proba.rename(columns={0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'})
        st.dataframe(
            df_prediction_proba,
            column_config={
                'Adelie': st.column_config.ProgressColumn(
                    'Adelie', format="%d %%", min_value=0, max_value=100
                ),
                'Chinstrap': st.column_config.ProgressColumn(
                    'Chinstrap', format="%d %%", min_value=0, max_value=100
                ),
                'Gentoo': st.column_config.ProgressColumn(
                    'Gentoo', format="%d %%", min_value=0, max_value=100
                ),
            },
            hide_index=True,
            width=704,
        )

# Display sidebar
render_sidebar()