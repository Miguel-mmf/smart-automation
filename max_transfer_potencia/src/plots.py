from matplotlib import pyplot as plt
import numpy as np


def create_plot(
    fitness_history,
    num_generations,
    save: bool = False,
):
    
    fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
    fitness_history_max = [np.max(fitness) for fitness in fitness_history]
    fitness_history_min = [np.min(fitness) for fitness in fitness_history]
    plt.figure(figsize=(15,5),facecolor="w")
    plt.plot(list(range(num_generations)), fitness_history_mean,'b', label = 'Fitness médio de cada geração', linewidth=2)
    plt.plot(list(range(num_generations)), fitness_history_max, 'g', label = 'Fitness máximo de cada geração', linewidth=2)
    plt.plot(list(range(num_generations)), fitness_history_min, 'r', label = 'Fitness mínimo de cada geração', linewidth=2)
    plt.legend()
    plt.grid()
    plt.title('Fitness através das gerações')
    plt.xlabel('Gerações')
    plt.ylabel(f'Fitness Máximo - {max(fitness_history_max):6.2f}')
    
    if save:
        plt.savefig('fitness_history.png')
        
    plt.show()