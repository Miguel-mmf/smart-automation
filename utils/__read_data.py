import pandas as pd


def read_data(
        path,
        usecols=None
    ):
    """This function reads the data from the path and returns a pandas dataframe.

    Args:
        path (str): The path to the data.

    Returns:
        DataFrame: A pandas dataframe.
    """

    try:
        if usecols:
            data = pd.read_parquet(path, columns=usecols)
        else:
            data = pd.read_parquet(path)
        print(f'{path}')
    except:
        file = path.split('/')[-1]
        if usecols:
            data = pd.read_parquet(f'UECEAR_SIEMENS/Modelos_Gemeo/data/{file}', columns=usecols)
        else:
            data = pd.read_parquet(f'UECEAR_SIEMENS/Modelos_Gemeo/data/{file}')
        print(f'UECEAR_SIEMENS/Modelos_Gemeo/data/{file}')
    

    data.plot(figsize=(16,12), subplots=True)
    
    return data