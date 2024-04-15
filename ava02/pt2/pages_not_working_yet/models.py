# %%
import json
import os  # to read environment variables
import time  # to simulate a real time data, time loop
from catboost import CatBoostRegressor
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
from datetime import datetime, date
from tensorflow import keras  # load model

# %%
config = json.load(open("./config.json", "r"))

# %%
st.set_page_config(
    page_title="Real-Time Control Panel",
    page_icon="✅",
    # initial_sidebar_state="collapsed",
    layout="wide",
)

# %%
@st.cache_resource
def load_keras_model(model_path):
    model = keras.models.load_model(model_path)
    return model


@st.cache_resource
def load_catboost_model(model_path):
    cat_model_reloaded = CatBoostRegressor()
    cat_model_reloaded.load_model(model_path)
    return cat_model_reloaded

# %%
@st.cache_data
def get_data() -> pd.DataFrame:
    today = date.today().strftime("%Y-%m-%d")
    if os.path.exists(f"models_page_data_{today}.csv"):
        return pd.read_csv(f"models_page_data_{today}.csv")
    else:
        now = datetime.now()
        df = pd.DataFrame(
            data=dict(
                time = [now], # .strftime("%H:%M:%S")
                set_point_freq = [30],
                set_point_angle = [45],
                frequency = [30],
                pt_k_1 = [0],
                pt_k_2 = [0],
                frequency_error = [1.2],
                angle = [45],
                angle_error = [0.5],
                prediction = [0],
                model = ['MLP']
            ),
            index=[now],
        )
        df.set_index("time", inplace=True)
        df.to_csv(f"models_page_data_{today}.csv")
        return df

# %%
df = get_data()

# dashboard title
st.title(config['models_page_title'])

with st.expander("Opcões de Simulação"):
    # top-level filters
    left_column_1, rigth_column_1 = st.columns(2, gap='large')
    with left_column_1:
        freq_filter = st.selectbox(
            "Selecione a frequência: ",
            [i for i in range(1, 60)],
            index=29
        )
        angle_filter = st.selectbox(
            "Selecione o ângulo: ",
            [i for i in range(0, 70)],
            index=45
        )

    with rigth_column_1:
        start_button = st.button("Start Simulation", key="start", type="primary", use_container_width=True)
        end_button = st.button("End Simulation", key="end", type="secondary", use_container_width=True)
        model_type = st.selectbox(
            "Selecione o modelo: ",
            ["CatBoost", "MLP"],
            index=1
        )
        
loaded_model = (
    load_keras_model(config['keras_model_path_to_pages'])
    if model_type == "MLP"
    else load_catboost_model(config['catboost_model_path_to_pages'])
)
num_inputs = 4 if model_type == "MLP" else 2

placeholder = st.empty()

last_kpi_values = None
if start_button:
    
    # near real-time / live feed simulation
    while True:
        
        start_time = time.time()
        
        if last_kpi_values is None:
            last_kpi_values = df.iloc[-1]
        
        now = datetime.now()
        
        a = np.asarray(df.iloc[-1][['set_point_freq', 'set_point_angle', 'pt_k_1', 'pt_k_2']], dtype=np.float32).reshape(1, -1)
        if model_type == "MLP":
            predicted_mca = loaded_model.predict(a)[0][0]
        else:
            predicted_mca = loaded_model.predict(a)[0]
        
        df.at[now,"time"] = now
        df.at[now,'set_point_freq'] = freq_filter
        df.at[now,'set_point_angle'] = angle_filter
        df.at[now, 'pt_k_1'] = last_kpi_values['prediction']
        df.at[now, 'pt_k_2'] = last_kpi_values['pt_k_1']
        df.at[now, 'frequency'] = last_kpi_values['set_point_freq']*np.random.uniform(0.98, 1.08)
        df.at[now, 'prediction'] = predicted_mca
        df.at[now, 'angle'] = angle_filter
        df.at[now, 'frequency_error'] = df.at[now, 'frequency'] - df.at[now, 'set_point_freq']*np.random.uniform(0.9, 1.1)
        df.at[now, 'angle_error'] = df.at[now, 'angle'] - df.at[now, 'set_point_angle']*np.random.uniform(0.9, 1.1)
        df.at[now, 'model'] = model_type
        # print(df)

        # creating KPIs
        last_row = df.iloc[-1]
        
        with placeholder.container():
            
            st.markdown("### Frequencia vs Tempo")
            fig = px.line(
                data_frame=df, x="time", y=["set_point_freq", "prediction"]
            )
            fig.update_layout(
                title="Frequência vs Tempo",
                xaxis_title="Tempo",
                yaxis_title="Frequência",
                legend_title="Frequência",
                height=500,
                # font=dict(
                #     family="Courier New, monospace",
                #     size=18,
                #     color="RebeccaPurple"
                # )
            )
            st.write(fig, use_container_width=True)

            st.markdown("### Dados em tempo real")
            st.dataframe(df, height=500, use_container_width=True, hide_index=True)
            time.sleep(config['delay'] - (time.time() - start_time))
            
            last_kpi_values = last_row
            
            removed_cols  = [
                col for col in df.columns
                if col not in config['columns']
            ]
            if removed_cols != []:
                df.drop(columns=removed_cols, inplace=True)
            df.to_csv(f"models_page_data_{date.today()}.csv")
        
        if end_button:
            end_button = False
            start_button = False
            st.markdown("### Detailed Data View")
            st.dataframe(df)
            # st.stop()