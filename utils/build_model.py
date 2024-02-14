from tensorflow import keras
from tensorflow.keras import layers
from functools import partial
from river.utils.math import minkowski_distance as md
from river import neighbors, tree
from river.utils import numpy2dict
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, max_error, mean_absolute_error, r2_score
import numpy as np

import tensorflow as tf
tf.keras.utils.set_random_seed(812)

from time import time
import mlflow

cp = configparser.ConfigParser()
cp.read('../../utils/credenciais-mlflow.ini')

# Credenciais e parâmetros do artifact store (MinIO)
minio_access_key = cp['minio']['AWS_ACCESS_KEY_ID']
minio_secret_key = cp['minio']['AWS_SECRET_ACCESS_KEY']
minio_endpoint = cp['minio']['MLFLOW_S3_ENDPOINT_URL']
artifact_url = cp['minio']['ARTIFACT_URL']

# Credenciais e parâmetros do back-end store (MySQL)
db_user = cp['mlflow-database']['db_user']
db_password = cp['mlflow-database']['db_password']
db_host = cp['mlflow-database']['db_host']
db_port = cp['mlflow-database']['db_port']
db_name = cp['mlflow-database']['db_name']

os.environ['AWS_ACCESS_KEY_ID'] = minio_access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = minio_secret_key
os.environ['MLFLOW_S3_ENDPOINT_URL'] = minio_endpoint

remote_server_uri = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
mlflow.set_tracking_uri(remote_server_uri)

def build_MLP_model(hp, input_shape=(5), dropout=True):
    """This function will build a sequential model with the hyperparameters.

    Args:
        hp (_type_): The hyperparameters.
        input_shape (tuple, optional): The input shape. Defaults to (5).

    Returns:
        keras_model: The model with the hyperparameters.
    """

    model = keras.Sequential(name='MLP_Model') #name='Sequential_Model'
    # model.add(layers.Dense(units=hp.Int('units', min_value=10, max_value=256, step=8), activation='relu', input_shape=(input_shape,)))
    model.add(
        layers.Dense(
            units = hp.Int('units', min_value=1, max_value=150, step=1),
            # activation = hp.Choice('activation', values=['relu','sigmoid','tanh']),
            activation = 'relu',
            input_shape=(input_shape,)
        )
    )

    if dropout:
        model.add(
            keras.layers.Dropout(hp.Float("dropout", min_value=0, max_value=1, step=0.001)) #(densex)
        )

    # Number of layers of the MLP is a hyperparameter.
    for i in range(hp.Int("mlp_layers", 1, 3)):
        # Number of units of each layer are different hyperparameters with different names.
        #model.add(layers.Dense(units=hp.Int(f"units_{i}", min_value=10, max_value=256, step=8), activation='relu'))
        model.add(
            layers.Dense(
                units=hp.Int(f"units_{i}", min_value=1, max_value=(150 - i), step=1),
                # activation = hp.Choice('activation', values=['relu','sigmoid','tanh'])
                activation = 'relu'
            )
        )
        if dropout:
            model.add(
                keras.layers.Dropout(hp.Float("dropout", min_value=0, max_value=1, step=0.001)) #(densex)
            )
        
    model.add(layers.Dense(units=1, activation = 'relu'))# hp.Choice('activation', values=['relu','sigmoid']))) # ,'tanh'

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate= hp.Float("lr", min_value=1e-6, max_value=1e-1, sampling="log")
        ),
        loss='mse',
        metrics=[
            'mae',
            # 'mse',
            keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
            keras.metrics.RootMeanSquaredError(name='root_mean_squared_error'),
            keras.losses.MeanAbsolutePercentageError(reduction="auto", name='mean_absolute_percentage_error')
        ]
    )
    
    return model

#======================================================================================================================================


def build_CFNN_model(hp, input_shape=(5), dropout=True):
    """This function will build a non-sequential model with the hyperparameters.

    Args:
        hp (_type_): The hyperparameters.
        input_shape (tuple, optional): The input shape. Defaults to (8).

    Returns:
        keras_model: The model with the hyperparameters.
    """

    inputs = keras.Input(shape=input_shape, name='Input_Layer')
    densex =  keras.layers.Dense(hp.Int('units', min_value=1, max_value=150, step=1), activation='relu', name='Hidden_Layer_1')(inputs)
    
    if dropout:
        densex = keras.layers.Dropout(hp.Float("dropout", min_value=0, max_value=1, step=0.001))(densex)

    # Number of layers of the MLP is a hyperparameter.
    for i in range(hp.Int("mlp_layers", 1, 2)):
        # Number of units of each layer are
        # different hyperparameters with different names.
        densex = keras.layers.Dense(hp.Int(f"units_{i}", min_value=1, max_value=150, step=1), activation='relu')(densex)
        
        if dropout:
            densex = keras.layers.Dropout(hp.Float("dropout", min_value=0, max_value=1, step=0.001))(densex)

    # dense2 =  keras.layers.Dense(hp.Int('units', min_value=10, max_value=256, step=8), activation='relu')(dense1)
    outputs = keras.layers.Dense(1, activation='relu')(densex)
    
    # Merge all available features into a single large vector via concatenation
    x = keras.layers.concatenate([inputs, outputs])
    output_final = keras.layers.Dense(
        1,
        activation='relu'
        # activation = hp.Choice('activation', values=['relu','sigmoid'])
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=output_final,
        name='CFNN_Model'
    )

    model.compile(
        optimizer = keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4] # hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
            )
        ),
        loss='mse',
        metrics=[
            'mae',
            # 'mse',
            keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
            keras.metrics.RootMeanSquaredError(name='root_mean_squared_error'),
            keras.losses.MeanAbsolutePercentageError(reduction="auto", name='mean_absolute_percentage_error')
        ]
    )
    
    return model


#======================================================================================================================================


def build_knn_model(n_neighbors, window_size, p, aggregation_method):
    """
        Construção do modelo de KNN com os hiperparametros definidos:
        Entradas: 
            - n_neighbors: número de vizinhos (int)
            - window_size: tamanho da janela (int)
            - p: parâmetro de distância (float)
            - aggregation_method: método de agregação 1: média, 2: mediana, 3: média ponderada (float)
        Saida:
            - Modelo de KNN com os hiperparametros definidos
    """
  
    agg_string = 'mean' if int(aggregation_method)==1 else 'median' if int(aggregation_method)==2 else 'weighted_mean'
    engine_knn = neighbors.LazySearch
    model = neighbors.KNNRegressor(
        n_neighbors=int(n_neighbors),
        aggregation_method=agg_string, 
        engine=engine_knn(window_size=int(window_size),
                        dist_func=partial(md, p=p))
        )
    return model


#======================================================================================================================================


def evaluate_knn(n_neighbors, window_size, p, aggregation_method, dataset_train, dataset_validation, experiment_id=None):
    """
        Treinamento e validação do KNN:
        Entradas: 
            - n_neighbors: número de vizinhos (int)
            - window_size: tamanho da janela (int)
            - p: parâmetro de distância (float)
            - aggregation_method: método de agregação 1: média, 2: mediana, 3: média ponderada (float)
            - dataset_train: dataset de treinamento (dicionario de arrays)
            - dataset_validation: dataset de validação (dicionario de arrays)
        Saida:
            - MSE do modelo de KNN com os hiperparametros definidos (float)
    """
    y_pred_list = []
    y_valid_list = []    
    ini = time()
    model = build_knn_model(n_neighbors, window_size, p, aggregation_method)
    for x, y in dataset_train:
        yi = y['y']
        y_p = model.predict_one(x)
        if yi != 0:
            if abs(100*(y_p - yi)/yi) >= 0.05:
                model.learn_one(x, yi)
        else:
            model.learn_one(x, yi)
    fim = time()

    for x, y in dataset_validation:
        yi = y['y']
        y_valid_list.append(yi)
        y_pred = model.predict_one(x)
        y_pred_list.append(y_pred)
    
    y_pred_array = np.array(y_pred_list)
    y_valid_array = np.array(y_valid_list)
    
    model = "KNN"
    params = {"n_neighbors":int(n_neighbors), "window_size":int(window_size), "p":p, "aggregation_method":aggregation_method}
    params['aggregation_method'] = 'mean' if int(aggregation_method)==1 else 'median' if int(aggregation_method)==2 else 'weighted_mean'
    MSE = round(mean_squared_error(y_valid_array, y_pred_array),4)
    MAE = round(mean_absolute_error(y_valid_array, y_pred_array),4)
    MAPE = round(mean_absolute_percentage_error(y_valid_array, y_pred_array)*100,2)
    MAXE = round(max_error(y_valid_array, y_pred_array),4)
    R2 = round(r2_score(y_valid_array, y_pred_array),4)
    TRAIN_TIME = fim - ini
    
    if experiment_id is not None:
        ############ REGISTRANDO NO MLFLOW ##################
        with mlflow.start_run(experiment_id=experiment_id):
            
            # Atributos do modelo
            mlflow.log_param("Modelo", model)
            mlflow.log_param("Hiperparametros", params)

            # Desempenho
            mlflow.log_metric("MSE", MSE)
            mlflow.log_metric("MAE", MAE)
            mlflow.log_metric("MAPE - pp", MAPE)
            mlflow.log_metric("MAXE", MAXE)
            mlflow.log_metric("R2", R2)
            mlflow.log_metric("Training Time - s", TRAIN_TIME)
            # for i in range(len(y_pred_array)):
            #     mlflow.log_metric("previsao", y_pred_array[i],step=i)

    return (-MSE)


#======================================================================================================================================


def build_linear_model(alpha, power_t):
    model = SGDRegressor(
        alpha=alpha,
        power_t=power_t,
        random_state=0
    )
    return model


#======================================================================================================================================


def evaluate_linear(alpha,power_t,x_train, x_valid, y_train, y_valid):
    y_pred_list = []
    y_valid_list = []
    model = build_linear_model(alpha, power_t)
    model.fit(x_train, y_train[0])
    y_pred_array = model.predict(x_valid)
    MSE = mean_squared_error(y_valid[0], y_pred_array)
    return (-MSE)


#======================================================================================================================================


def build_cat_model(params):
    iterations=params['iterations'] 
    depth=params['depth']
    l2=params['l2'] 
    bagging_temperature=params['bagging_temperature'] 
    random_strength=params['random_strength']
    
    model = CatBoostRegressor(
                            iterations=int(iterations),
                            depth=int(depth),
                            l2_leaf_reg=l2,
                            bagging_temperature=bagging_temperature,
                            random_strength=random_strength,
                            random_seed=13
                            )
    
    return model


#======================================================================================================================================


def evaluate_tree(iterations, depth, l2, bagging_temperature, random_strength, train_set, eval_set):

    X_train = train_set[0]
    Y_train = train_set[1]

    x_ev = eval_set[0]
    y_ev = eval_set[1]
    
    params = {"iterations":iterations, 
              "depth":depth, 
              "l2":l2, 
              "bagging_temperature":bagging_temperature, 
              "random_strength":random_strength}
    
    model = build_cat_model(params)
    
    model.fit(X_train,Y_train,verbose=0)
    
    pred = model.predict(x_ev)
    
    MSE = mean_squared_error(pred,y_ev)
    MAPE = mean_absolute_percentage_error(pred,y_ev)*100
    MAXE = max_error(pred,y_ev)
    return (-MSE)


#=============================================================================================================================================


def build_model(params):

    n_inputs = params['n_inputs'] 
    n_outputs = params['n_outputs']
    dropout = params['dropout']
    l2 = params['l2']
    n_layers = int(params['n_layers'])
    n_neurons = params['n_neurons']
    
    dp_hp = tf.keras.layers.Dropout(dropout)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(n_inputs,)))
    
    for i in range(n_layers):
    #print(i)
        layer_hp = int(n_neurons[i])
        model.add(dp_hp)
        model.add(tf.keras.layers.Dense(layer_hp,
                                        kernel_regularizer=tf.keras.regularizers.L2(l2),
                                        activation='relu'))
        model.add(tf.keras.layers.Dense(n_outputs, activation='relu'))
        model.compile(loss='mean_squared_error',
                  optimizer= 'Adam',
                  metrics=["mse","mae","mape"])
    return model


#=============================================================================================================================================

def evaluate_network(lay1,lay2,train_set, eval_set,lay3=0,lay4=0):

    X_train = train_set[0]
    Y_train = train_set[1]

    x_ev = eval_set[0]
    y_ev = eval_set[1]

    n_inputs = len(X_train.columns) 
    n_outputs = len(Y_train.columns)

    dropout = 0.001
    l2 = 0.0001
    
    n_neurons = [lay1,lay2,lay3,lay4, 0]
    n_layers = 0
    
    for i in range(len(n_neurons)):
        if n_neurons[i] == 0:
            n_layers = i

    params = {'dropout':dropout, 'l2':l2, 'n_layers':n_layers, 'n_neurons':n_neurons, 'n_inputs':n_inputs, 'n_outputs':n_outputs}
    
    
    model = build_model(params)
    #monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
    #patience=PATIENCE, verbose=0, mode='auto',
    #                        restore_best_weights=True)

    
    model.fit(X_train,Y_train,validation_data=eval_set,verbose=0,epochs=250)
    #epochs = monitor.stopped_epoch
    #epochs_needed.append(epochs)
    
    # Predict on the out of boot (validation)
    #pred = model.predict(eval_set[0])
    
    
    scores = model.evaluate(eval_set[0],eval_set[1])
    MSE = scores[1]
    MAE = scores[2]
    MAPE = scores[3]
    #MAXE = max_error(eval_set[1],pred)
    
    tf.keras.backend.clear_session()
    return (-MSE)


#=============================================================================================================================================

def build_mlp(params, n_inputs, n_outputs):

    lay1 = params['lay1']
    lay2 = params['lay2']
    try:
        lay3 = params['lay3']
    except:
        lay3 = 0
    try:
        lay4 = params['lay4']
    except:
        lay4 = 0

    dropout = 0.001
    l2 = 0.0001
    
    n_neurons = [lay1,lay2,lay3,lay4, 0]
    n_layers = 0
    
    for i in range(len(n_neurons)):
        if n_neurons[i] == 0:
            n_layers = i
            break

    params = {'dropout':dropout, 'l2':l2, 'n_layers':n_layers, 'n_neurons':n_neurons, 'n_inputs':n_inputs, 'n_outputs':n_outputs}
    
    model = build_model(params)
    
    return model
