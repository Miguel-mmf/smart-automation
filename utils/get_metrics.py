import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, max_error, r2_score
import pandas as pd


def erapm(real_data, predicted_data):
    """This function will calculate the average relative percentage error.

    Args:
        real_data (_type_): _description_
        predicted_data (_type_): _description_
    """

    absolute_errors = abs(real_data - predicted_data)
    relative_errors = (absolute_errors / real_data) * 100
    max_relative_error = np.round(relative_errors.max().max(),4)
    print(f"Erro Relativo Percentual Médio: {max_relative_error}")

    return max_relative_error
    

def erpm(real_data, predicted_data):
    """This function will calculate the absolute max percentage error.

    Args:
        real_data (_type_): _description_
        predicted_data (_type_): _description_
    """
    # Erro relativo percentual maximo
    
    absolute_percentage_errors = abs((real_data - predicted_data) / real_data) * 100
    max_relative_error = np.round(absolute_percentage_errors.max().max(),4)
    
    return max_relative_error


def erp_std(real_data, predicted_data):
    """This function will calculate the absolute max percentage error.

    Args:
        real_data (_type_): _description_
        predicted_data (_type_): _description_
    """
    # Erro relativo percentual maximo
    
    absolute_percentage_errors = abs((real_data - predicted_data) / real_data) * 100
    std = np.std(absolute_percentage_errors)
    
    return std


def get_metrics(real_data, predicted_data):

    print(f"MAE: {np.round(mean_absolute_error(real_data,predicted_data), 4)}")
    print(f"MSE: {np.round(mean_squared_error(real_data,predicted_data), 4)}")
    print(f"RMSE: {np.round(np.sqrt(mean_squared_error(real_data,predicted_data)), 4)}")
    print(f"MAPE: {np.round(mean_absolute_percentage_error(real_data,predicted_data)*100, 4)}")
    print(f"ERPM: {np.round(erpm(real_data,predicted_data), 4)}")
    # return np.round(mean_absolute_percentage_error(real_data,predicted_data)*100, 4)


#====================================================================================================================

def gera_metricas(df,nome_real,nome_previsto, verbose=False):
    """
        Recebe um dataframe contendo uma coluna com os valores reais (nome_real) e uma coluna com os valores preditos
        (nome_previsto) e os nomes dessas colunas.
        
        Retorna um dicionário contendo as métricas calculadas e um dataframe contendo os pontos nas duas curvas cujo
        erro absoluto é máximo.
        
        Métricas calculadas:
        RMSE: Raíz quadrada do erro médio quadrático. 
        MSE: Erro médio quadrático.
        MAE: Erro médio absoluto.
        MAXE: Máximo erro absoluto.
        MAXPE: Erro máximo percentual absoluto - Valor percentual do máximo erro absoluto.
        Dessa forma, o ponto utilizado para calcular essa métrica é o mesmo ponto do valor máximo absoluto.
        MAPE: Erro médio percentual. Desconsidera valores abaixo de 1 do valor real da variável de saída. Isso evita que
        o valor dessa métrica tenda a infinito por conta de pontos onde a variável de saída tende a zero.
        R2: Coeficiente de determinação.
    """

    df_out = df.copy()
    df_out['dif_abs'] = abs(df_out[nome_real]-df_out[nome_previsto])
    

    # MSE: Erro médio quadrático.
    mse = mean_squared_error(df_out[nome_real], df_out[nome_previsto])

    # RMSE: Raíz quadrada do erro médio quadrático.
    rmse = np.sqrt(mse)

    # MAE: Erro médio absoluto
    mae = mean_absolute_error(df_out[nome_real], df_out[nome_previsto])
    
    # MAXE: Máximo erro absoluto.
    maxe = max_error(df_out[nome_real], df_out[nome_previsto])

    # R2: Coeficiente de determinação.
    r2 = r2_score(df_out[nome_real], df_out[nome_previsto])
    
    # MAPE: Erro médio percentual.
    df_out = df[df[nome_real]>=1]
    mape = mean_absolute_percentage_error(df_out[nome_real], df_out[nome_previsto])*100

    # MAXPE: Erro máximo percentual absoluto - Valor percentual do máximo erro absoluto.  
    maxdif = pd.DataFrame()
    maxdif[nome_previsto] = df_out[df_out['dif_abs']==df_out['dif_abs'].max()][nome_previsto]
    maxdif[nome_real] = df_out[df_out['dif_abs']==df_out['dif_abs'].max()][nome_real]
    
    m1 = maxdif[nome_previsto].values
    m2 = maxdif[nome_real].values
    max_pe = (m1[0]-m2[0])/m2[0]
    max_pe = abs(max_pe)*100

    metricas = {"RMSE":rmse,
                "MSE":mse,
                "MAE":mae,
                "MAXE":maxe,
                "MAXPE":max_pe,
                "MAPE":mape,
                "R2":r2 }

    if verbose==True:
        for k in metricas.keys():
            print(f'{k}: {metricas[k]: .3f}')
        
    return metricas, maxdif

#====================================================================================================================