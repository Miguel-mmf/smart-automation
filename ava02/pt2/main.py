# %%
import json
import sys
import os  # to read environment variables
import time  # to simulate a real time data, time loop
from catboost import CatBoostRegressor
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
from datetime import datetime, date
from tensorflow import keras  # load model

sys.path.append("ava02/pt2/utils/")
from utils import (
    create_fuzzy,
    get_results,
    calc_error,
    calc_delta_error,
    calc_new_frequency,
    scale_data
)

# %%
config = json.load(open("config.json", "r"))

# %%
st.set_page_config(
    page_title="Real-Time Control Panel",
    page_icon="✅",
    initial_sidebar_state="collapsed",
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
    if os.path.exists(f"data_{today}.csv"):
        return pd.read_csv(f"data_{today}.csv")
    else:
        now = datetime.now()
        df = pd.DataFrame(
            data=dict(
                time = [now], # .strftime("%H:%M:%S")
                pressure_set_point = [30],
                pressure_predicted = [0],
                error = [0],
                delta_error = [0],
                frequency = [0],
                delta_frequency = [0],
                pt_k_1 = [0],
                pt_k_2 = [0],
                # frequency_error = [1.2],
                angle_set_point = [45],
                angle = [45],
                angle_error = [0.5],
                model = ['MLP']
            ),
            index=[now],
        )
        df.set_index("time", inplace=True)
        df.to_csv(f"data_{today}.csv")
        return df

# %%
df = get_data()
FS = create_fuzzy()

# dashboard title
st.title(config['main_page_title'])

with st.expander("Opcões de Simulação"):
    # top-level filters
    left_column_1, rigth_column_1 = st.columns(2, gap='large')
    with left_column_1:
        pressure_set_point = st.selectbox(
            "Selecione o valor desejado de pressão: ",
            [i for i in range(4,18)],
            index=2
        )
        angle_filter = st.selectbox(
            "Selecione o ângulo da válvula: ",
            [i for i in range(0, 70, 3)],
            index=10
        )

    with rigth_column_1:
        start_button = st.button("Iniciar a simulação", key="start", type="primary", use_container_width=True)
        end_button = st.button("Encerrar a simulação", key="end", type="secondary", use_container_width=True)
        model_type = st.selectbox(
            "Selecione o modelo: ",
            ["CatBoost", "MLP"],
            index=1
        )
        
loaded_model = (
    load_keras_model(config['keras_model_path'])
    if model_type == "MLP"
    else load_catboost_model(config['catboost_model_path'])
)
num_inputs = 4 if model_type == "MLP" else 2


# creating a single-element container
def create_metrics_container(df):
    placeholder = st.empty()

    with placeholder.container():

        # create three columns
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6, gap='small')

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Pressão desejada 📡",
            value=df["pressure_set_point"].values[0],
            delta=0,
        )
        
        kpi2.metric(
            label='Pressão 📡',
            value=df["pressure_predicted"].values[0],
            delta=0
        )
        
        kpi3.metric(
            label='Frequência 📡',
            # value=df["frequency_error"].values[0],
            value=df["frequency"].values[0],
            delta=0
        )
        
        kpi4.metric(
            label='Delta Frequência 📡',
            value=df["delta_frequency"].values[0],
            delta=0
        )
        
        kpi5.metric(
            label='Ângulo 📐',
            value=df["angle"].values[0],
            delta=0
        )
        
        kpi6.metric(
            label='Delta Ângulo 📐',
            value=df["angle_error"].values[0],
            delta=0
        )
    
    return placeholder


placeholder = create_metrics_container(df)

last_kpi_values = None
if start_button:
    
    placeholder.empty()
    start_button = False
    # near real-time / live feed simulation
    while True:
        
        start_time = time.time()
        
        if last_kpi_values is None:
            last_kpi_values = df.iloc[-1]
        
        #Normalizando os dados
        _, x_scaler = scale_data(df[['frequency', 'angle_set_point', 'pt_k_1', 'pt_k_2']], scaler_type='minmax')
        _, y_scaler = scale_data(df[['pressure_predicted']], scaler_type='minmax')
        
        x_scaled = scale_data(df[['frequency', 'angle_set_point', 'pt_k_1', 'pt_k_2']], x_scaler)
        # Conferir
        if model_type == "MLP":
            nor = np.array([[2581.45792141, 2543.96720659,  770.72376701,  769.92793663]])#, 771.6440832 ]])
            x_scaled = df[['frequency', 'angle_set_point', 'pt_k_1', 'pt_k_2']].div(nor)
        a = np.asarray(x_scaled.iloc[-1], dtype=np.float32).reshape(1, -1) if model_type == "MLP" else np.asarray(df.iloc[-1][['frequency', 'angle_set_point', 'pt_k_1', 'pt_k_2']], dtype=np.float32).reshape(1, -1)
        
        if model_type == "MLP":
            pressure_predicted = loaded_model.predict(a)[0][0] * 771.6440832
            #pressure_predicted = y_scaler['pressure_predicted'].inverse_transform(loaded_model.predict(a))[0][0]
        else:
            pressure_predicted = loaded_model.predict(a)[0]
        
        now = datetime.now()
        error = calc_error(pressure_set_point, pressure_predicted)
        delta_error = calc_delta_error(error, last_kpi_values['error'])
        
        df.at[now,"time"] = now
        df.at[now,'pressure_set_point'] = pressure_set_point
        df.at[now,'angle_set_point'] = angle_filter
        df.at[now, 'pt_k_1'] = last_kpi_values['pressure_predicted']
        df.at[now, 'pt_k_2'] = last_kpi_values['pt_k_1']
        df.at[now, 'pressure_predicted'] = pressure_predicted
        df.at[now, 'error'] = error
        df.at[now, 'delta_error'] = delta_error
        df.at[now, 'angle'] = angle_filter*np.random.uniform(0.96, 1.04)
        # df.at[now, 'frequency_error'] = df.at[now, 'frequency'] - df.at[now, 'pressure_set_point']*np.random.uniform(0.9, 1.1)
        df.at[now, 'angle_error'] = df.at[now, 'angle'] - df.at[now, 'angle_set_point']*np.random.uniform(0.99, 1.01)
        df.at[now, 'model'] = model_type
        # print(df)
        
        # fuzzy logic
        error = max(min(error, 15), -15)
        delta_error = max(min(delta_error, 5), -5)
        df.at[now, 'delta_frequency'] = get_results(error, delta_error, FS)
        new_frequency = calc_new_frequency(last_kpi_values['frequency'], df.at[now, 'delta_frequency'])
        # A frequencia deve ser limitada
        df.at[now, 'frequency'] = (
            new_frequency
            if ((new_frequency < 60) and (new_frequency > 30))
            else 30 if (new_frequency < 30)
            else 60 if (new_frequency > 60)
            else 30
        )

        # creating KPIs
        last_row = df.iloc[-1]

        with placeholder.container():
            
            with st.expander("Métricas", expanded=True):
                
                # create three columns
                kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6, gap='small')

                # fill in those three columns with respective metrics or KPIs
                kpi1.metric(
                    label="Pressão desejada 📡",
                    value=round(last_row["pressure_set_point"], 2),
                    delta=round(last_row["pressure_set_point"] - last_kpi_values["pressure_set_point"], 2)
                )
                
                kpi2.metric(
                    label='Pressão 📡',
                    value=round(last_row["pressure_predicted"], 2),
                    delta=round(last_row["pressure_predicted"] - last_kpi_values["pressure_predicted"], 2)
                )
                
                kpi3.metric(
                    label='Frequência 📡',
                    value=round(last_row["frequency"], 2),
                    delta=round(last_row["frequency"] - last_kpi_values["frequency"], 2)
                )
                
                kpi4.metric(
                    label='Delta Frequência 📡',
                    value=round(last_row["delta_frequency"], 2),
                    delta=round(last_row["delta_frequency"] - last_kpi_values["delta_frequency"], 2)
                )
                
                kpi5.metric(
                    label='Ângulo 📐',
                    value=round(last_row["angle"], 2),
                    delta=round(last_row["angle"] - last_kpi_values["angle"], 2)
                )
                
                kpi6.metric(
                    label='Delta Ângulo 📐',
                    value=round(last_row["angle_error"], 2),
                    delta=round(last_row["angle_error"] - last_kpi_values["angle_error"], 2)
                )
            
            
            # create two columns for charts
            fig_col1, fig_col2 = st.columns(2)
            with fig_col1:
                st.markdown("### Pressão vs Tempo")
                # fixar uma janela de tempo para apresentacao dos dados
                fig = px.line(
                    data_frame=df,
                    x="time",
                    y=["pressure_set_point", "pressure_predicted"],
                    color_discrete_map={"pressure_set_point": "blue", "pressure_predicted": "red"},
                    labels={"pressure_set_point": "Desejada", "pressure_predicted": "Predita"},
                )
                fig.update_layout(
                    height=400,
                    width=580,
                    xaxis_title="Tempo",
                    yaxis_title="Pressão M.C.A",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                )
                st.write(fig)
            
            with fig_col2:
                st.markdown("### Erros vs Tempo")
                fig = px.line(
                    data_frame=df,
                    x="time",
                    y=["delta_error", "error", "delta_frequency"],
                    color_discrete_map={"delta_error": "blue", "error": "red", "delta_frequency": "green"},
                    labels={"delta_error": "Delta Erro", "error": "Erro", "delta_frequency": "Delta Frequência"}
                )
                fig.update_layout(
                    height=400,
                    width=580,
                    xaxis_title="Tempo",
                    yaxis_title="Pressão M.C.A",
                    legend=dict(
                        title="Legenda",
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    xaxis=dict(
                        showline=True,
                        showgrid=True,
                        showticklabels=True,
                        linecolor='rgb(204, 204, 204)',
                        linewidth=2,
                        ticks='outside',
                        tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                        ),
                    ),
                )
                st.write(fig)
            
            
            st.markdown("### Frequencia e Angulo vs Tempo")
            fig = px.line(
                data_frame=df,
                x="time",
                y=["frequency", "angle_set_point", "angle"],
            )
            fig.update_layout(
                height=400,
                width=1200,
                xaxis_title="Tempo",
                yaxis_title="Frequência (Hz) | Ângulo (°)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
            )
            st.write(fig)

            st.markdown("### Dados em tempo real")
            st.dataframe(df, height=500, use_container_width=True, hide_index=True)
            time.sleep(config['delay'] - (time.time() - start_time) if (config['delay'] - (time.time() - start_time))>0 else config['delay'])
        
        last_kpi_values = last_row
        
        removed_cols  = [
            col for col in df.columns
            if col not in config['columns']
        ]
        if removed_cols != []:
            df.drop(columns=removed_cols, inplace=True)
        df.to_csv(f"data_{date.today()}.csv")
        
        if end_button:
            end_button = False
            placeholder = create_metrics_container(df)

            st.markdown("### Detailed Data View")
            st.dataframe(df)
            # st.stop()
            
            # RETIRAR ISSO DEPOIS
            os.remove(f"data_{date.today()}.csv")