from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
try:
    from create_circuit import calc_gain_circuit
except ImportError as e:
    print(e)
    from src import calc_gain_circuit
    from src import get_best_individual


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
    plt.plot(list(range(num_generations)), data,'b', label = 'Fitness mínimo de cada geração', linewidth=2)
    plt.legend()
    plt.grid()
    plt.title(f'Fitness {method.capitalize()} através das gerações')
    plt.xlabel('Gerações')
    
    if save:
        plt.tight_layout()
        # plt.savefig(f'fitness_{method}_history_{problem}.png')
        plt.savefig(f'fitness_{method}_history_{problem}.pdf')
    
    # plt.show()


def create_Is_plot(
    fitness_history,
    population_history,
    save: bool = False,
    problem: str = 'p1'
):
    gain_values = [
        calc_gain_circuit(
            Vin=22,
            Vs=1,
            f=1,
            R1=individual[0],
            R2=individual[1],
            RC=individual[2],
            RE=individual[3],
            C=individual[4],
            CE=individual[5],
            type='DC',
            return_Is_current=False,
        )
        for individual in population_history
    ]
    is_values = [
        np.min(fit_value) - (2000 - gain)**2
        for fit_value, gain in zip(fitness_history, gain_values)
    ]
    
    plt.figure(figsize=(15,5), facecolor="w")
    plt.plot(
        list(range(len(fitness_history))),
        is_values,
        color='blue',
        label = 'Is',
        linewidth=3
    )
    plt.text(
        x=len(fitness_history),
        y=np.max(is_values)*0.1,
        s=f'{round(np.min(is_values), 2)} A',
        fontsize=12,
        horizontalalignment='right',
        verticalalignment='center',
        fontweight='bold',
    )
    plt.grid()
    plt.legend()
    plt.title('Comportamento da Is na otimição')
    plt.xlabel('Gerações')
    plt.ylabel('Corrente Is')
    
    if save:
        plt.tight_layout()
        # plt.savefig(f'./ava03/pt1/images/evolution_components_AG/{component_name}_history_{problem}.png')
        plt.savefig(f'./ava03/pt1/images/evolution_components_AG/Is_history_{problem}.pdf')
    
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
    
    for problem in tqdm(['p1', 'p2'], desc='Problemas'):
        
        try:
            results = json.load(open(f'result_{problem}.json'))
        except FileNotFoundError as e:
            print(e)
            results = json.load(open(f'./ava03/pt1/result_{problem}.json'))
            
    #     values = results['fitnessHistory']
        
    #     create_plot(
    #         values,
    #         save=True,
    #         method='min',
    #         problem=problem
    #     )
        
    #     values_without_large_fitness = [np.min(fitness) for fitness in values if np.min(fitness) < 100**2]
    #     create_single_fitness_plot(
    #         values_without_large_fitness,
    #         len(values_without_large_fitness),
    #         save=True,
    #         method='min',
    #         problem=problem
    #     )
        component_names = ['R1','R2','RC','RE','C', 'CE']
        best_individual_history = np.array(results["bestIndividualHistory"])
        population_history = np.array(results["populationHistory"])
        for component_index in tqdm(range(best_individual_history.shape[1]), desc='Componentes'):
            tqdm.write(f'Componente: {component_names[component_index]}', end=' - ')
            create_best_individual_evolution_plot(
                best_individual_history[:, component_index],
                component_names[component_index],
                best_individual_history.shape[0],
                save=True,
                problem=problem
            )
            
            create_population_evolution_plot(
                population_history[:, :, component_index],
                best_individual_history[:, component_index],
                component_names[component_index],
                population_history.shape[0],
                save=True,
                problem=problem
            )
            tqdm.write('Componente plotado')
    
        if problem == 'p2':
            create_Is_plot(
                results['fitnessHistory'],
                best_individual_history,
                save=True,
                problem=problem
            )