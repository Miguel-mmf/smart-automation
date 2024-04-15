import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer


def scale_data(df, scalers=None, scaler_type='minmax'):
    """
    Scales a dataset using either MinMaxScaler, StandardScaler or QuantileTransformer from scikit-learn.

    Parameters:
    df (pandas.DataFrame): The dataset to be scaled.
    scaler_type (str): The type of scaler to use. Either 'minmax', 'standard' or 'quantile'.

    Test:
    for s in scalers.values():
        print(s.get_params(True), end=' | ')
        print(s.data_min_, end=' | ')
        print(s.data_max_)

    Returns:
    pandas.DataFrame: The scaled dataset.
    dict: A scalers dict.
    """
    # Adicionei a condição para usar o escalador, caso ja haja um. Realiza fit_transform apenas com os dados de treinamento e depois apenas o transfrm nos dados de validação e treinamento.
    # Ajustei o range dos dados no minmax para dar uma folga, caso dodos posteriores tenham maior módulo os dados normalizados ainda permanecerão entre 0 e 1.
    # Sugestão: Dar a opção de normalizar apenas dados de entrada.

    df_norm = pd.DataFrame()
    
    if scalers:
        
        for col in df.columns:
            feat = np.array(df[col])
            aux = feat.shape
            data = scalers[col].transform(feat.reshape(-1,1)) # f.transform(feat)
            df_norm[col] = data.reshape(aux)
            
        return df_norm

    else: # shift nas linhas abaixo
    
        scalers = dict()
        for col in df.columns:
            
            if scaler_type == 'minmax':
                scaler = MinMaxScaler(feature_range=(0.15,0.85)) # default
            elif scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'quantile':
                scaler = QuantileTransformer()
            else:
                raise ValueError("Invalid scaler type. Must be 'minmax', 'standard' or 'quantile'.")
                
            scalers[col] = scaler
            feat = np.array(df[col])
            aux = feat.shape
            data = scalers[col].fit_transform(feat.reshape(-1,1)) # f.transform(feat)
            df_norm[col] = data.reshape(aux)
    
        return df_norm, scalers


def get_original_data(df_norm, scalers):
    """
    Back-projection to the original dataset space using either MinMaxScaler, StandardScaler or QuantileTransformer from scikit-learn.

    Parameters:
    df_norm (pandas.DataFrame): The scaled dataset.
    scalers (str): The type of scaler to use. Either 'minmax', 'standard' or 'quantile'.

    Returns:
    pandas.DataFrame: The original dataset.
    """
    
    df = pd.DataFrame()
            
    for col in df_norm.columns:
        scaler = scalers[col]
        print(scaler)
        feat = np.array(df_norm[col])
        aux = feat.shape
        data = scaler.inverse_transform(feat.reshape(-1,1)) # f.transform(feat)
        print(data[0])
        df[col] = data.reshape(aux)

    return df
