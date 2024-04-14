import streamlit as st
import pandas as pd

rules_path = '../../ava02/pt1/regras.csv'

def rules():
    st.title('Rules')
    
    data = pd.read_csv(rules_path)
    st.write(data)
    

if __name__ == '__main__':
    rules()