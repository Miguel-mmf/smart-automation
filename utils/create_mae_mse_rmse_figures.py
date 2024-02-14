import matplotlib.pyplot as plt
from matplotlib import rcParams 
plt.style.use('default')
rcParams.update(
    {
        'figure.figsize':(16,5),
        'figure.facecolor':'white',
        'font.size':'16',
        'axes.grid': True,
        'axes.grid.axis': 'both',  
        # 'text.usetex' : True,
        # 'font.family' : 'serif', 
        # 'font.serif' : 'cm'   
    }
)


def create_mae_mse_rmse_figures(df_history, save=False, filename=' ', return_fig=False):
    """This function creates the MAE, MSE and RMSE figures for the training and validation sets.

    Args:
        df_history (DataFrame): The history of the training and validation sets.
        save (bool, optional): The flag to save the figures. Defaults to False.
        filename (str, optional): The filename to save the figures. Defaults to ' '.
    """

    cols = df_history.columns
    length_df_history = len(df_history)
    metrics = {
        'mae': 'mean_absolute_error',
        'mse': 'mean_squared_error',
        'rmse': 'root_mean_squared_error',
        # 'mape': 'mean_absolute_percentage_error',
    }

    figs = list()
    axss = list()

    for key, value in metrics.items():
        
        fig, axs = plt.subplots(1,1,figsize=(12,5),dpi=600)
        
        if key in cols:
            axs.plot(df_history[key],lw=2,color='red', label=f'Training {key.upper()}')
            axs.plot(df_history[f'val_{key}'],lw=2,color='blue', label=f'Validation {key.upper()}')
        else:
            try:
                axs.plot(df_history[value],lw=2,color='red', label=f'Training {key.upper()}')
                axs.plot(df_history[f'val_{value}'],lw=2,color='blue', label=f'Validation {key.upper()}')
            except Exception as e:
                print(e)

        axs.grid(True)
        axs.set_ylabel(key.upper(), fontsize=22)
        axs.set_xlabel('Epochs', fontsize=22)
        axs.set_xlim([0, length_df_history+1])
        # axs.set_xticks([i for i in range(0,length_df_history,2)], fontsize=20)
        axs.legend(fontsize=18)
        axs.grid(visible=True)

        figs.append(fig)
        axss.append(axs)
        
        if save:
            if filename == ' ':
                f = key.upper()
            else:
                f = filename + '_' + key.upper()
            #plt.savefig(f'{filename}.eps', format='eps', dpi=600)
            plt.savefig(f'{f}.png', format='png', dpi=600)
            #plt.savefig(f'{filename}.pdf', format='pdf', dpi=600) 

    if save:
        if filename == ' ':
            f = 'history.csv'
        else:
            f = filename + 'history.csv'

        df_history.to_csv(f)

    if return_fig:
        return figs, axss