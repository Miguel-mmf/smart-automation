import mlflow

class MyKNN(mlflow.pyfunc.PythonModel):
    """This subclass of PythonModel uses a KNN model to make predictions.

    Args:
        mlflow (_type_): A MLflow model.
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.

        Args:
            context (_type_): MLflow context where the model artifact is stored.
        """        

        import pickle
        with open(context.artifacts['knn_model_path'], 'rb') as f:
            model = pickle.load(f)

        self.model = model
        print('O modelo foi carregado com sucesso!')
        print(f'Tipo do modelo: {type(self.model)}')


    def predict(self, context, model_input):
        """This function....

        Args:
            context (_type_): _description_
            model_input (_type_): _description_

        Returns:
            _type_: _description_
        """                
        
        print('Inicio da validação...')
        y_pred_list = list()
        for x, y in model_input:
            y_p = self.model.predict_one(x)
            y_pred_list.append(y_p)
        
        print('Fim da validação. Retornando resultados...')
        return y_pred_list