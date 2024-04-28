import json
import humps
import numpy as np
from tqdm import tqdm
from time import time
from datetime import date

from fitness import calc_fitness


def binary_to_decimal(binary):
    """This function converts a binary number to a decimal number.

    Args:
        binary (_type_): _description_

    Returns:
        int: The decimal number.
    """
    if isinstance(binary, list):
        binary = ''.join(map(str, binary))
    elif isinstance(binary, np.ndarray):
        binary = ''.join(map(str, binary.tolist()))
    else:
        try:
            binary = str(binary)
        except Exception as e:
            raise ValueError(f'Cannot convert {type(binary)} to str: {e}') from e
    
    size = len(binary)
    return sum(int(binary[size-i-1])*(2**i) for i in range(size))


def initial_solutions(
    num_ind: int,
    verbose: bool = False
) -> tuple:
    """
    Generates initial binary solutions for a population.

    Args:
        num_ind (int): The number of individuals in the population.
        verbose (bool): Whether to print information about the generated population.

    Returns:
        tuple: A tuple containing the initial binary population.
    """

    pop_size = (num_ind, num_ind)
    initial_population = np.random.randint(2, size = pop_size)

    if verbose:
        print(f'Tamanho da População = {pop_size}')
        print(f'População Inicial: \n{initial_population}')

    return initial_population


def elapsed_time(func):
    
    """
    Decorator to measure the elapsed time of a function.

    Args:
        func: The function to be measured.

    Returns:
        The result of the function.
"""
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f'Time elapsed: {round(time() - start, 2)}s')
        return result
    return wrapper


def validate_patience(patience, num_generations):
    """
    Validates and sets the patience value for early stopping.

    Args:
        patience (int | bool): The patience value for early stopping.
        num_generations (int): The total number of generations.

    Returns:
        int | bool: The validated patience value.
    """
    if not patience:
        patience = num_generations
        print(f'Patience set to {patience}.')
    return patience


def check_patience(curr_generation, patience, fitness_history):
    """
    Checks if early stopping criteria are met based on patience and fitness history.

    Args:
        curr_generation (int): The current generation number.
        patience (int): The patience value for early stopping.
        fitness_history (list): List of fitness values over generations.

    Returns:
        bool: True if early stopping criteria are met, False otherwise.
    """

    if curr_generation > patience and np.all(np.max(fitness_history[-patience]) == np.max(fitness_history[-1])):
        tqdm.write(f'Early stopping at generation {curr_generation+1}')
        return True
    return False


def get_best_neihbor_solution(population, fitness, method):
    """
    Returns the best individual from a population based on fitness and method.

    Args:
        population (numpy.ndarray): The population of individuals.
        fitness (numpy.ndarray): The fitness values of individuals.
        method (str): The method to determine the best individual, either 'max' or 'min'.

    Returns:
        numpy.ndarray: The best individual from the population.
    """
    if method == 'max':
        select_func = np.argmax
    elif method == 'min':
        select_func = np.argmin

    return population[select_func(fitness)], select_func(fitness)


def save_results(filename, result):
    """
    Saves a dictionary result to a JSON file.

    Args:
        filename (str): The name of the file to save the result to.
        result (dict): The dictionary result to be saved.
    """
    with open(filename, 'w') as f:
        json.dump(result, f, indent=4)


def mount_results(
    first_solution,
    best_iteration,
    best_solution,
    best_solution_fitness,
    solution_history,
    solution_weight_history,
    solution_value_history,
    best_solution_history,
    solution_fitness_history,
    best_solution_fitness_history,
    tabu_list_history,
    method,
    patience
) -> dict:
    
    result = {
        "first_solution": first_solution.tolist(),
        'best_iteration': best_iteration,
        'best_solution': best_solution.tolist(),
        'best_solution_fitness': float(best_solution_fitness),
        'best_solution_history': [
            best_solution.tolist()
            for best_solution in best_solution_history
        ],
        'solution_weight_history': [
            weight.tolist()
            for weight in solution_weight_history
        ],
        'solution_value_history': [
            value.tolist()
            for value in solution_value_history
        ],
        'best_solution_fitness_history': [
            fitness.tolist()
            for fitness in best_solution_fitness_history
        ],
        'solution_history': [
            [s.tolist() for s in solution]
            for solution in solution_history
        ],
        'solution_fitness_history': [
            fitness.tolist()
            for fitness in solution_fitness_history
        ],
        'tabu_list_history': tabu_list_history,
        'method': method,
        'patience': patience,
    }
    result = humps.camelize(result)

    return result

def show_verbose_info(iteration, best_solution, best_solution_fitness, tabu_list, verbose=False):
    if verbose:
        print(f'Iteração {iteration}', end=' ')
        print(f'Melhor Indivíduo: {best_solution}', end=' ')
        print(f'Fitness do Melhor Indivíduo: {best_solution_fitness}', end=' ')
        print(f'Lista Tabu: {tabu_list}')


def init_historical_data(population, weight, value, threshold):
    (
        first_fitness,
        total_weight_neighbors,
        total_value_neighbors
    ) = calc_fitness(weight, value, population, threshold)
    best_solution = population[np.argmax(first_fitness)]
    best_solution_fitness = np.max(first_fitness)
    tabu_list = []
    solution_history = [population]
    solution_weight_history = [total_weight_neighbors]
    solution_value_history = [total_value_neighbors]
    best_solution_history = [best_solution]
    solution_fitness_history = [first_fitness]
    best_solution_fitness_history = [best_solution_fitness]
    tabu_list_history = []
    
    return (
        best_solution,
        best_solution_fitness,
        tabu_list,
        solution_history,
        solution_weight_history,
        solution_value_history,
        best_solution_history,
        solution_fitness_history,
        best_solution_fitness_history,
        tabu_list_history
    )


def update_best_solution(
    best_neighbor_fitness,
    best_solution,
    best_neighbor,
    best_solution_fitness,
    best_iteration,
    iteration
) -> None:
    if best_neighbor_fitness > best_solution_fitness:
        best_solution = best_neighbor
        best_solution_fitness = best_neighbor_fitness
        best_iteration = iteration


def main_optimize(
    num_iterations: int,
    weight: np.ndarray,
    value: np.ndarray,
    threshold: float,
    tabu_list_size: int,
    fitness_func: callable,
    method: str,
    patience: int | bool,
    solution_history: list,
    solution_weight_history: list,
    solution_value_history: list,
    best_solution_history: list,
    solution_fitness_history: list,
    best_solution_fitness_history: list,
    tabu_list_history: list,
    best_solution: np.ndarray,
    best_solution_fitness: float,
    tabu_list: list,
    verbose: bool = False
):
    
    best_iteration = 0
    print('Inicializando Busca Tabu...')
    for iteration, _ in enumerate(tqdm(range(num_iterations), desc='Progresso: '), start=1):
        
        show_verbose_info(
            iteration=iteration,
            best_solution=best_solution,
            best_solution_fitness=best_solution_fitness,
            tabu_list=tabu_list,
            verbose=verbose
        )

        neighbors = []
        for pos in tqdm(range(10), desc=f'Vizinhos ({iteration}):', leave=True):
            neighbor = best_solution.copy()
            neighbor[pos] = 1 - neighbor[pos] if pos not in tabu_list else neighbor[pos]
            neighbors.append(
                np.array(neighbor, dtype=int)
                # if binary_to_decimal(neighbor) not in tabu_list
                if pos not in tabu_list
                else np.ones(10, dtype=int)
            )

        (
            fitness_neighbors,
            total_weight_neighbors,
            total_value_neighbors
        ) = fitness_func(weight, value, np.array(neighbors), threshold)

        solution_history.append(neighbors)
        solution_fitness_history.append(fitness_neighbors)
        solution_weight_history.append(total_weight_neighbors)
        solution_value_history.append(total_value_neighbors)
        
        best_neighbor, best_neighbor_fitness = get_best_neihbor_solution(
            neighbors,
            fitness_neighbors,
            method
        )

        # tabu_list.append(binary_to_decimal(best_neighbor))
        tabu_list.append(int(np.argmax(fitness_neighbors)))
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)
        tabu_list_history.append(tabu_list.copy())

        update_best_solution(
            best_neighbor_fitness,
            best_neighbor,
            best_solution,
            best_solution_fitness,
            best_iteration,
            iteration
        )
            
        best_solution_fitness_history.append(best_solution_fitness)
        best_solution_history.append(best_solution)
        
        if check_patience(iteration, patience, best_solution_fitness_history):
            break
    
    return (
        best_solution,
        best_solution_fitness,
        best_iteration,
        tabu_list,
        solution_history,
        solution_weight_history,
        solution_value_history,
        best_solution_history,
        solution_fitness_history,
        best_solution_fitness_history,
        tabu_list_history
    )

@elapsed_time
def tabu_search(
    population: np.ndarray,
    num_iterations: int,
    weight: np.ndarray,
    value: np.ndarray,
    threshold: float,
    tabu_list_size: int,
    fitness_func: callable = calc_fitness,
    method: str = 'max',
    patience: int | bool = 200,
    verbose: bool = False,
    filename: str = f'tabu_search_results_{date.today()}.json'
) -> tuple:

    (
        best_solution,
        best_solution_fitness,
        tabu_list,
        solution_history,
        solution_weight_history,
        solution_value_history,
        best_solution_history,
        solution_fitness_history,
        best_solution_fitness_history,
        tabu_list_history
    ) = init_historical_data(population, weight, value, threshold)
    show_verbose_info(
        iteration=0,
        best_solution=best_solution,
        best_solution_fitness=best_solution_fitness,
        tabu_list=tabu_list,
        verbose=verbose
    )

    (
        best_solution,
        best_solution_fitness,
        best_iteration,
        tabu_list,
        solution_history,
        solution_weight_history,
        solution_value_history,
        best_solution_history,
        solution_fitness_history,
        best_solution_fitness_history,
        tabu_list_history
    ) = main_optimize(
        num_iterations,
        weight,
        value,
        threshold,
        tabu_list_size,
        fitness_func,
        method,
        patience,
        solution_history,
        solution_weight_history,
        solution_value_history,
        best_solution_history,
        solution_fitness_history,
        best_solution_fitness_history,
        tabu_list_history,
        best_solution,
        best_solution_fitness,
        tabu_list,
        verbose
    )
    
    result = mount_results(
        population[0],
        best_iteration,
        best_solution,
        best_solution_fitness,
        solution_history,
        solution_weight_history,
        solution_value_history,
        best_solution_history,
        solution_fitness_history,
        best_solution_fitness_history,
        tabu_list_history,
        method,
        patience
    )
    
    save_results(filename, result)

    return best_solution, best_solution_fitness #, solution_history, solution_fitness_history


if __name__ == '__main__':
    import os
    start = time()
    weight = np.array([10,8,2,4,15,5,3,1,12,9])
    value = np.array([1200,200,300,1000,1500,800,2000,40,500,3000])
    
    best_trial_solution = None
    best_trial_solution_fitness = 0
    best_trial_solution_fitness_history = []
    filename = ''
    patience = 40
    
    for trial in tqdm(range(100), desc='Trials: '):
        
        f = f'./ava03/pt2/busca_tabu/tabu_search_trials/{trial}_tabu_search_results_{date.today()}.json'
        population = initial_solutions(weight.shape[0], False)
        
        (
            best_solution,
            best_solution_fitness,
            # solution_history,
            # solution_fitness_history
        ) = tabu_search(
            population=population,
            num_iterations=1000,
            weight=weight,
            value=value,
            threshold=35,
            tabu_list_size=2,
            method='max',
            patience=30,
            filename=f,
            verbose=False
        )
        
        # print(f'Melhor Indivíduo: {best_solution}')
        # print(f'Fitness do Melhor Indivíduo: {best_solution_fitness}')
        
        if best_solution_fitness > best_trial_solution_fitness:
            best_trial_solution = best_solution
            best_trial_solution_fitness = best_solution_fitness
            filename = f
        best_trial_solution_fitness_history.append(best_solution_fitness)
        
        if (
            len(best_trial_solution_fitness_history) > patience
            and best_trial_solution_fitness_history[-patience] == best_trial_solution_fitness_history[-1]
        ):
            break\
    
    for trial in tqdm(range(100), desc='Removendo Arquivos de Resultados: '):
        try:
            f = f'./ava03/pt2/busca_tabu/tabu_search_trials/{trial}_tabu_search_results_{date.today()}.json'
            if (
                os.path.exists(f)
                and filename != f
            ):
                os.remove(f)
        except FileNotFoundError as e:
            print(e)
    
    print(f'Melhor Solução: {best_trial_solution}')
    print(f'Fitness da Melhor Solução: {best_trial_solution_fitness}')
    print(f'Arquivo de Resultados: {filename}')
    print(f'Tempo Total: {round(time() - start, 2)}s')
    print('Fim do Programa')