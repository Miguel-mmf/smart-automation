import os
import mlflow
import configparser


def experiment_names_equal(experiments, expected_names):
    actual_names = [e.name for e in experiments if e.name != "Default"]
    return actual_names == expected_names #, (actual_names, expected_names)


def connect_mlflow(
        init_file='../config/credenciais-mlflow.ini'
    ):
    """_summary_

    Args:
        init_file (str, optional): This file contains the credentials to connect to the artifact store (MinIO) and to the back-end store (MySQL). Defaults to '../config/credenciais-mlflow.ini'.
    """

    cp = configparser.ConfigParser()
    cp.read(init_file)

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

    # return mlflow


def get_experiment_id(experiment_name):

    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    return experiment_id


def get_run_id(experiment_id, run_name):

    run_id = mlflow.search_runs(experiment_id=experiment_id, filter_string=f"tags.mlflow.runName = '{run_name}'").iloc[0].run_id

    return run_id


def create_experiment(
        experiment_name,
        init_file='../config/credenciais-mlflow.ini'
    ):
    """This function creates a new MLflow Experiment.

    Args:
        experiment_name (_type_): _description_
        init_file (str, optional): _description_. Defaults to '../config/credenciais-mlflow.ini'.
    """
    
    experiments = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")

    cp = configparser.ConfigParser()
    cp.read(init_file)

    # Credenciais e parâmetros do artifact store (MinIO)
    artifact_url = cp['minio']['ARTIFACT_URL']
    print(experiment_names_equal(experiments, [experiment_name]))
    if not experiment_names_equal(experiments, [experiment_name]):
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_url
    )
    
        # Create a new MLflow Experiment
        mlflow.set_experiment(experiment_id=experiment_id)
    

def write_artifact(
        experiment_name,
        run_name,
        artifact_path,
        artifact_file,
        artifact_type='csv',
        init_file='./config/credenciais-mlflow.ini'
    ):
    """This function writes the artifact to the MLflow server.

    Args:
        experiment_name (_type_): _description_
        run_name (_type_): _description_
        artifact_path (_type_): _description_
        artifact_file (_type_): _description_
        artifact_type (str, optional): _description_. Defaults to 'csv'.
        init_file (str, optional): _description_. Defaults to './config/credenciais-mlflow.ini'.
    
    Example:
        write_artifact(
            experiment_name='Quickstart - Miguel',
            run_name='grandiose-shrimp-306',
            artifact_path='models/TG1_POT/results/concatenated.csv',
            artifact_file='models/TG1_POT/results/concatenated.csv',
            artifact_type='csv',
            init_file='./config/credenciais-mlflow.ini'
        )

    Returns:
        _type_: _description_
    """

    connect_mlflow(init_file)
    experiment_id = get_experiment_id(experiment_name)
    # run_id = get_run_id(experiment_id, run_name)

    with mlflow.start_run(experiment_id=experiment_id):
        if artifact_type == 'csv':
            mlflow.log_artifact(artifact_file, artifact_path)
        elif artifact_type == 'model':
            mlflow.sklearn.log_model(artifact_file, artifact_path)
        else:
            print('Tipo de artefato não suportado.')
            return None


def write_logs(
        experiment_name,
        model_name,
        model_param,
        list_of_log_dict,
        init_file='../config/credenciais-mlflow.ini'
    ):
    """This function writes the logs of the model to the MLflow server.

    Args:
        experiment_name (_type_): _description_
        model_name (_type_): _description_
        model_param (_type_): _description_
        list_of_log_dict (_type_): _description_
    
    Example:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

        write_logs(
            experiment_name='Quickstart - Miguel',
            model_name='Basic CFNN model for TV_TEMP data',
            model_param=best_params['params'],
            list_of_log_dict=[
                {
                    'MSE': mean_squared_error(y_test,preds),
                },
                {
                    'MAE': mean_absolute_error(y_test,preds),
                },
                {
                    'MAPE': mean_absolute_percentage_error(y_test,preds)
                }
            ],
            init_file='./config/credenciais-mlflow.ini'
        )
    """

    connect_mlflow(init_file)
    experiment_id = get_experiment_id(experiment_name)

    if experiment_id is not None:

        with mlflow.start_run(experiment_id=experiment_id):
            
            # Atributos do modelo
            # mlflow.log_param("Modelo", model_name)
            # mlflow.log_param("Hiperparametros", model_param)

            for log_dict in list_of_log_dict:
                
                mlflow.log_metric(
                    tuple(*log_dict.items())[0],
                    tuple(*log_dict.items())[1]
                )


def write_real_and_predicted_values(
        experiment_name,
        model_name,
        model_param,
        y_pred_array=None,
        y_true_array=None,
        init_file='../config/credenciais-mlflow.ini'
    ):
    """This function writes the real and predicted values of the model to the MLflow server.

    Args:
        experiment_name (_type_): _description_
        model_name (_type_): _description_
        model_param (_type_): _description_
        y_pred_array (_type_): _description_
        y_true_array (_type_): _description_
    
    Example:
        write_real_and_predicted_values(
            experiment_name='MLflow Quickstart - Miguel',
            model_name='Basic CFNN model for TV_TEMP data',
            model_param=best_params['params'],
            y_pred_array=preds.reshape(1,-1),
            y_true_array=y_scalers['Saida.T'].inverse_transform(preds).reshape(1,-1),
            init_file='./config/credenciais-mlflow.ini'
        )
    """

    connect_mlflow(init_file)
    experiment_id = get_experiment_id(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id):
        print(y_pred_array)
        print(y_true_array)

        # Atributos do modelo
        mlflow.log_param("Modelo", model_name)
        mlflow.log_param("Hiperparametros", model_param)

        # é possível usar threads para enviar o real e previsto ao mesmo tempo? diminuiria o tempo de execução pela metade?
        for i in range(len(y_pred_array)): # dá para inserir o tqdm aqui para auxiliar na visualização do progresso
            if isinstance(y_pred_array, list):
                mlflow.log_metric("previsao", y_pred_array[i], step=i)
            if isinstance(y_true_array, list):
                mlflow.log_metric("real", y_true_array[i], step=i)
            if y_true_array: mlflow.log_metric("real", y_true_array[i], step=i)