import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def create_best_individual_evolution_plot(
    component_data: list,
    component_name: str,
    num_generations: int,
    save: bool = False,
    problem: str = 'p1'
):
    plt.figure(figsize=(15,5), facecolor="w")
    plt.plot(
        list(range(num_generations)),
        component_data,
        color='blue',
        label = component_name,
        linewidth=3
    )
    plt.legend()
    plt.grid()
    plt.xlabel('Gerações')
    plt.ylabel('Resitência' if component_name.startswith('R') else 'Capacitância')
    plt.title(f'Evolução de {component_name} por geração')
    
    if save:
        plt.tight_layout()
        # plt.savefig(f'./ava03/pt1/images/evolution_components_AG/{component_name}_history_{problem}.png')
        plt.savefig(f'./ava03/pt1/images/evolution_components_AG/{component_name}_history_{problem}.pdf')
    
    # plt.show()


def create_population_evolution_plot(
    component_data: list,
    best_individual_data: list,
    component_name: str,
    num_generations: int,
    save: bool = False,
    problem: str = 'p1'
):
    plt.figure(figsize=(15,5), facecolor="w")
    for x in range(num_generations):
        plt.scatter(
            x=[x]*component_data.shape[1],
            y=component_data[x],
            # color='blue',
            cmap='viridis',
            s=13
        )

    plt.plot(
        list(range(num_generations)),
        best_individual_data,
        color='red',
        label = f'Best {component_name}',
        linewidth=3
    )
    plt.legend()
    plt.grid()
    plt.xlabel('Gerações')
    plt.ylabel('Resitência' if component_name.startswith('R') else 'Capacitância')
    plt.title(f'Evolução de {component_name} por geração')
    
    if save:
        plt.tight_layout()
        # plt.savefig(f'./ava03/pt1/images/evolution_components_AG/{component_name}_history_{problem}.png')
        plt.savefig(f'./ava03/pt1/images/evolution_components_AG/{component_name}_history_{problem}.pdf')
    
    # plt.show()


def create_single_fitness_plot(
    data: list,
    num_generations: int,
    save: bool = False,
    method: str = 'min',
    problem: str = 'p1'
):
    plt.figure(figsize=(15,5),facecolor="w")
    plt.plot(list(range(num_generations)), data,'b', label = 'Fitness máximo de cada geração', linewidth=2)
    plt.legend()
    plt.grid()
    plt.title(f'Fitness {method.capitalize()} através das gerações')
    plt.xlabel('Gerações')
    
    if save:
        plt.tight_layout()
        # plt.savefig(f'fitness_{method}_history_{problem}.png')
        plt.savefig(f'fitness_{method}_history_{problem}.pdf')
    
    # plt.show()


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
    if method == 'max':
        plt.ylabel(f'Fitness {method.capitalize()} - {max(fitness_history_max):6.2f}')
    elif method == 'min':
        plt.ylabel(f'Fitness {method.capitalize()} - {min(fitness_history_min):6.2f}')
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
        
    # plt.show()


if __name__ == '__main__':
    
    import json
    # problem = 'p2'
    
    # for problem in tqdm(['p1', 'p2'], desc='Problemas'):
        
    try:
        results = json.load(open('tabu_search_results_2024-04-28.json'))
    except FileNotFoundError as e:
        print(e)
        results = json.load(open('./ava03/pt2/tabu_search_results_2024-04-28.json'))
    # tabu_search_results_2024-04-27.json
    values = results['solutionFitnessHistory']
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # ax.set_title('Fitness over time')
    
    # ax.plot(
    #     [results['bestSolutionFitness']]*len(results['solutionFitnessHistory']),
    #     label='best_fits',
    #     color='red',
    #     linewidth=2
    # )
    # for i in range(len(results['solutionFitnessHistory'])):
    #     ax.scatter(
    #         x=[i for _ in range(len(results['solutionFitnessHistory'][i]))],
    #         y=results['solutionFitnessHistory'][i],
    #         marker='o'
    #     )

    # ax.legend()
    # plt.show()
    
    
    create_plot(
        values,
        save=True,
        method='max',
    )
    
    # values_without_large_fitness = [np.max([f for f in fitness if f != 0]) for fitness in values]
    # create_single_fitness_plot(
    #     values_without_large_fitness,
    #     len(values_without_large_fitness),
    #     save=True,
    #     method='max',
    # )
    
    # create_population_evolution_plot(
    #     np.array(results['solutionHistory']),
    #     np.array(results['bestSolutionFitness']*len(results['solutionHistory'])),
    #     'R',
    #     len(results['solutionHistory']),
    #     save=True,
    # )