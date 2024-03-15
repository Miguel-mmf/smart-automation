import pandas as pd
import streamlit as st
import streamlit as st
from examples.senusoid import Senusoid

def main():
    st.title('Gerador Automático de Regras Fuzzy')
    
    with st.container(border=True):
        st.write('### Valores de entrada')
        col1, col2 = st.columns(2)

        with col1:
            internal_example = st.selectbox('Exemplos internos', ['Onda Senoidal'], placeholder="Choose an option")
            uploaded_file = st.file_uploader('Upload de arquivo CSV')
        
        with col2:
            if internal_example and not uploaded_file:
                sin = Senusoid()
                st.line_chart(sin.get_values[['y']])
            elif uploaded_file:
                # bytes_data = uploaded_file.getvalue()
                # st.write(bytes_data)
                dataframe = pd.read_csv(uploaded_file, sep=';', decimal='.')
                st.write(dataframe)

    if uploaded_file:
       st.line_chart(dataframe[['POTENCIA', 'POTENCIA(K-1)', 'POTENCIA(K+1)']])
    
    # Get user inputs
    input_variable = st.text_input('Enter input variable:')
    output_variable = st.text_input('Enter output variable:')
    rule = st.text_input('Enter rule:')
    
    # Perform fuzzy logic calculations
    # $PLACEHOLDER$ - Add code to perform fuzzy logic calculations based on user inputs
    
    # Display results
    st.write('Results:')
    st.write('Input variable:', input_variable)
    st.write('Output variable:', output_variable)
    st.write('Rule:', rule)
    # $PLACEHOLDER$ - Add code to display fuzzy logic results
    
if __name__ == '__main__':
    main()