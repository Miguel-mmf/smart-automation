from pathlib import Path

import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split, ShuffleSplit#, KFold, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import pandas as pd
import numpy as np

from datetime import datetime

import tensorflow as tf
from bayes_opt import BayesianOptimization
tf.keras.utils.set_random_seed(812)




#=============================================================================================================================================

def split_data(df, pct = [0.7,0.1]):
    """
        Separa os dados de treinamento, validação e teste em ordem cronológica.
        Obs: Os dados devem estar organizados em ordem cronológica.
        pct é uma lista com o percentual dos dados a serem separados no tipo:
        pct = [0.7,0.1] -> 70% dos dados serão reservados para treinamento, 10%
        dos dados serão reservados para validação e 20% para testes.
    """
    
    train_set = int(len(df)*pct[0])
    val_set = int(len(df)*(pct[0]+pct[1]))
    
    df_train = df.iloc[:train_set,:].copy()
    df_ev =  df.iloc[train_set:val_set,:].copy()
    df_test =  df.iloc[val_set:,:].copy()
    
    return df_train, df_ev, df_test

#=============================================================================================================================================

def make_features(df,outputs, inputs=None):
    """
        Recebe um dataframe, uma lista de outputs e uma lista opcional de inputs.
        Retorna dois dataframes contendo separadamente os inputs e outputs passados.
        Caso não sejam passados os inputs, a função simplesmente retorna o dataframe
        passado, exceto os outputs, como inputs.
    """
    
    x_out = pd.DataFrame()
    y_out = pd.DataFrame()
    
    for out in outputs:
        y_out[out] = df[out].copy()
    
    if inputs != None:
        for inp in inputs:
            x_out[inp] = df[inp].copy()
    
    else:
        for inp in df.columns:
            if inp not in outputs:
                x_out[inp] = df[inp].copy()
        
    return x_out, y_out

#=============================================================================================================================================

def normalize_inputs(df,outputs):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_norm = pd.DataFrame()
    for i in df.columns:
        if i not in outputs:
            feat = np.array(df[i])
            aux = feat.shape
            feat = feat.reshape(-1,1)
            f = scaler.fit(feat)
            data = f.transform(feat)
            data = data.reshape(aux)
            df_norm[i] = data
    df_norm[outputs] = df[outputs].copy()
    return df_norm

#=============================================================================================================================================

def std_fig(x,y_r,y_p,std):
    sd = np.max(std)
    sd = f"Desvio padrão máximo: {sd: .2f}"
    plt.figure()
    plt.grid()
    plt.plot(x,y_r, label='Real')
    plt.plot(x,y_p, label='Previsto')
    plt.fill_between(x,y_p-std,y_p+std, alpha=0.5, label='std')
    #plt.fill_between(x,y_p- 3*std,y_p+ 3*std, alpha=0.5, label='3 std')
    plt.xlabel('Tempo')
    plt.ylabel('Potência de saída (MW)')
    #plt.title('Previsão com intervalo de confiança')
    plt.legend()
    plt.tick_params('x', labelbottom=False)
    plt.grid(color='gray',linestyle='--',linewidth=0.5, axis='y')
    plt.grid(visible=False, axis='x')
    plt.annotate(sd,
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom')
    #plt.annotate('Pico', [20., 0.95])
    #plt.grid()
    figure = plt.gcf()
    figure.set_size_inches(4*1.6, 4)
    
    return figure

#=============================================================================================================================================

def f_bayes_opt(funcao,pbounds,init_points=10,n_iter=20):
    """
        Recebe uma função, os limites das variáveis de entrada, o número de iterações
        iniciais e o número de iterações de exploração. O pbounds é um dicionário 
        contendo as variáveis de entrada da função a ser otimizada e seus respectivos 
        limites. A função passada precisa retornar o valor de uma métrica a ser 
        otimizada. Exemplo -MAPE (negativo, pois o MAPE vai de 0-1, sendo o valor 
        ótimo, 0, assim, otimizar a métrica significa fazê-la tender a zero. Como é 
        utilizada a função para maximizar o resultado, é colocado o sinal de menos).
        Para exemplo de utilização, ver notebook "Exemplo_bayesian_optimization".
    """

    start = datetime.now()
    
    optimizer = BayesianOptimization(
        f=funcao,
        pbounds=pbounds,
        verbose=1,  # verbose = 1 prints only when a maximum
        # is observed, verbose = 0 is silent
        random_state=1,
        allow_duplicate_points=False)
    
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    stop = datetime.now()
    delta = stop-start
    
    print('Tempo decorrido: ', str(delta))
    print("Parâmetros ótimos: \n", optimizer.max)

    return optimizer.max

#=============================================================================================================================================

def gera_log(df_out, real_name, pred_name, file_path):
    """
        Salva a previsão do modelo para a variável de saída.
        Recebe o dataframe com a variável de saída e a previsão do 
        modelo, o nome da variável de saída, o nome do modelo
        e o nome do arquivo de saída.
    """
    
    log = file_path
    
    if log.name.endswith('.csv') == False:
        raise Exception('Extensão inválida! O nome do arquivo deve estar no formato "nome.csv".')
    if log.is_file():
        df = pd.read_csv(log, index_col='Unnamed: 0')
        if real_name not in df_out.columns:
            raise Exception('A variável de saída não pertence ao documento escolhido!')
    else:
        df = pd.DataFrame()
        df[real_name] = df_out[real_name].copy()

    df[pred_name] = df_out[pred_name].copy()
    try:
        df.to_csv(log)
        print(f"Arquivo salvo em {log}!")
    except:
        print("Erro ao salvar arquivo!")
    
    
    return df

#=============================================================================================================================================

def save_best_params(params,file_path):
    file = file_path
    if file.name.endswith('.csv') == False:
        raise Exception('Extensão inválida! O nome do arquivo deve estar no formato "nome.csv".')
    df = pd.DataFrame(params, index=[0])
    print(df.head())
    try:
        df.to_csv(file)
        print(f"Arquivo salvo em {file}!")
    except:
        print("Erro ao salvar arquivo!")
    
