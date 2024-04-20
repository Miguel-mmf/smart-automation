from matplotlib import pyplot as plt
import numpy as np


def create_single_fitness_plot(
    data: list,
    num_generations: int,
    save: bool = False,
    method: str = 'min',
    problem: str = 'p1'
):
    plt.figure(figsize=(15,5),facecolor="w")
    plt.plot(list(range(num_generations)), data,'b', label = 'Fitness mínimo de cada geração', linewidth=2)
    plt.legend()
    plt.grid()
    plt.title(f'Fitness {method.capitalize()} através das gerações')
    plt.xlabel('Gerações')
    
    if save:
        plt.tight_layout()
        plt.savefig(f'fitness_{method}_history_{problem}.png')
        plt.savefig(f'fitness_{method}_history_{problem}.pdf')
    
    plt.show()


def create_plot(
    fitness_history,
    save: bool = False,
    method: str = 'min',
    problem: str = 'p1'
):
    
    fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
    fitness_history_max = [np.max(fitness) for fitness in fitness_history]
    fitness_history_min = [np.min(fitness) for fitness in fitness_history if np.min(fitness) < 100**2]
    plt.figure(figsize=(15,5),facecolor="w")
    plt.plot(list(range(len(fitness_history_mean))), fitness_history_mean,'b', label = 'Fitness médio de cada geração', linewidth=2)
    plt.plot(list(range(len(fitness_history_max))), fitness_history_max, 'g', label = 'Fitness máximo de cada geração', linewidth=2)
    plt.plot(list(range(len(fitness_history_min))), fitness_history_min, 'r', label = 'Fitness mínimo de cada geração', linewidth=2)
    plt.legend()
    plt.grid()
    plt.title('Fitness através das gerações')
    plt.xlabel('Gerações')
    if method == 'min':
        plt.ylabel(f'Fitness {method.capitalize()} - {min(fitness_history_min):6.2f}')
    elif method == 'max':
        plt.ylabel(f'Fitness {method.capitalize()} - {max(fitness_history_max):6.2f}')
    else:
        raise ValueError('Método inválido. Escolha entre "max" ou "min"')
    
    if save:
        plt.tight_layout()
        plt.savefig(f'fitness_history_{problem}.png')
        plt.savefig(f'fitness_history_{problem}.pdf')
        create_single_fitness_plot(
            fitness_history_min,
            len(fitness_history_min),
            save=True,
            method=method,
            problem=problem
        )
        # create_single_fitness_plot(fitness_history_max, num_generations, save=True, method=method)
        
    plt.show()


if __name__ == '__main__':
    
    import json
    # problem = 'p1'
    for problem in ['p1', 'p2']:
        values = json.load(open(f'result_{problem}.json'))['fitnessHistory']
        
        create_plot(
            values,
            save=True,
            method='min',
            problem=problem
        )
        
        values_without_large_fitness = [np.min(fitness) for fitness in values if np.min(fitness) < 100**2]
        create_single_fitness_plot(
            values_without_large_fitness,
            len(values_without_large_fitness),
            save=True,
            method='min',
            problem=problem
        )