import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_absolute_percentage_error


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