import pandas as pd
import matplotlib.pyplot as plt


def difference_plot(real_data, predicted_data, filename=' ', save=False, return_fig=False):
    """This function will create a figure with the real and predicted power and difference area in another plot.

    Args:
        real_data (_type_): _description_
        predicted_data (_type_): _description_
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), dpi=600, sharex=True)

    axs[0].plot(real_data, lw=2, color='Red', marker='s', markersize=3)
    axs[0].plot(predicted_data, lw=2, color='Blue')

    axs[1].plot(real_data - predicted_data, lw=2, color='red')
    axs[1].fill_between(
        range(0, len(predicted_data)),
        (real_data - predicted_data).ravel(),
        lw=2,
        color='red',
        alpha=0.5
    )

    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].set_ylabel('Potência (MW)')
    axs[1].set_ylabel('Potência (MW)')
    axs[0].set_xlabel('Pontos')
    axs[1].set_xlabel('Pontos')
    axs[0].legend(['Potência Real','Potência Estimada'])
    axs[1].legend(['Diferença entre Potência Real e Estimada'])
    
    plt.tight_layout()

    if save:
        if filename == ' ':
            f = key.upper()
        else:
            f = filename + '_' + key.upper()
        #plt.savefig(f'{filename}.eps', format='eps', dpi=600)
        plt.savefig(f'{f}.png', format='png', dpi=600)
        #plt.savefig(f'{filename}.pdf', format='pdf', dpi=600)

    if return_fig:
        return fig, axs