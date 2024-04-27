import json
import humps
import numpy as np
from tqdm import tqdm
from time import time
from .core import crossover, mutation, selection
from .fitness import calc_fitness

def mount_circuit_results(parameters):
    """
    Returns a dictionary containing optimized circuit parameters.

    Args:
        parameters (list): A list of circuit parameters to be optimized.

    Returns:
        dict: A dictionary with optimized circuit parameters including 'Vs', 'Vin', 'f', 'R1', 'R2', 'RC', 'RE', 'C', 'CE'.
    """

    return {
        'Vs': 1,
        'Vin': 22,
        'f': 1,
        'R1': round(parameters[0][0], 2),
        'R2': round(parameters[0][1], 2),
        'RC': round(parameters[0][2], 2),
        'RE': round(parameters[0][3], 2),
        'C': round(parameters[0][4], 2),
        'CE': round(parameters[0][5], 2),
    }

def get_best_individual(population, fitness, method):
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
        fitness_idx = np.where(fitness == np.max(fitness))
    elif method == 'min':
        fitness_idx = np.where(fitness == np.min(fitness))

    return population[fitness_idx[0][0], :]

def validate_num_parents_offsprings(population, num_parents, num_offsprings):
    """
    Validates the number of parents and offsprings in a population.

    Args:
        population (numpy.ndarray): The population of individuals.
        num_parents (int): The number of parent individuals.
        num_offsprings (int): The number of offspring individuals.

    Raises:
        ValueError: If the sum of num_parents and num_offsprings is not equal to the population size.
    """
    if num_offsprings + num_parents != population.shape[0]:
        raise ValueError('The number of offsprings add and the number of parents must be equal to the population size.')

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

    if curr_generation > patience and np.all(fitness_history[-patience] == fitness_history[-1]):
        tqdm.write(f'Early stopping at generation {curr_generation+1}')
        return True
    return False

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
    parameters,
    fitness_history,
    population_history, 
    best_individual_history,
    method, num_generations,
    num_parents,
    num_offsprings,
    crossover_rate,
    mutation_rate,
    patience,
    population
) -> dict:
    """
    Mounts a dictionary result containing circuit parameters, fitness history, and optimization details.

    Args:
        parameters (list): The optimized circuit parameters.
        fitness_history (list): List of fitness values over generations.
        population_history (list): List of population states over generations.
        best_individual_history (list): List of best individuals over generations.
        method (str): The optimization method used, either 'max' or 'min'.
        num_generations (int): The total number of generations.
        num_parents (int): The number of parent individuals.
        num_offsprings (int): The number of offspring individuals.
        crossover_rate (float): The rate of crossover in the genetic algorithm.
        mutation_rate (float): The rate of mutation in the genetic algorithm.
        patience (int): The patience value for early stopping.
        population (numpy.ndarray): The population of individuals.

    Returns:
        dict: A dictionary containing circuit parameters, fitness history, and optimization details.
    """
    
    result = {'circuit': mount_circuit_results(parameters)}
    result['fitness_last_generation'] = [round(f,2) for f in fitness_history[-1]]
    result['parameters'] = parameters[0].tolist()
    result['method'] = method
    result['num_generations'] = num_generations
    result['num_parents'] = num_parents
    result['num_offsprings'] = num_offsprings
    result['crossover_rate'] = crossover_rate
    result['mutation_rate'] = mutation_rate
    result['patience'] = patience
    result['population'] = population.tolist()
    result['fitness_history'] = fitness_history
    result['population_history'] = population_history
    result['best_individual_history'] = best_individual_history

    result = humps.camelize(result)

    return result

def get_max_or_min_fitness(fitness, method):
    """
    Determines the index of the maximum or minimum fitness value.

    Args:
        fitness (numpy.ndarray): The fitness values.
        method (str): The method to determine the index, either 'max' or 'min'.

    Returns:
        numpy.ndarray: The index of the maximum or minimum fitness value.
    """
    if method == 'max':
        return np.where(fitness == np.max(fitness))
    elif method == 'min':
        return np.where(fitness == np.min(fitness))
    else:
        raise ValueError('Método inválido. Escolha entre "max" ou "min"')
    
def main_optimize(
    population, 
    num_generations,
    num_parents,
    num_offsprings,
    crossover_rate,
    mutation_rate,
    fitness_func: callable = calc_fitness,
    method: str = 'min',
    patience: int | bool = 200,
    verbose: bool = False,
):
    """
    Main function to optimize a population using a genetic algorithm.

    Args:
        population (numpy.ndarray): The initial population of individuals.
        num_generations (int): The total number of generations.
        num_parents (int): The number of parent individuals.
        num_offsprings (int): The number of offspring individuals.
        crossover_rate (float): The rate of crossover in the genetic algorithm.
        mutation_rate (float): The rate of mutation in the genetic algorithm.
        fitness_func (callable): The fitness function to evaluate individuals.
        method (str): The optimization method, either 'max' or 'min'.
        patience (int | bool): The patience value for early stopping.
        verbose (bool): Whether to display verbose output.

    Returns:
        tuple: A tuple containing the final population and fitness history.
    """
    fitness_history = []
    best_individual_history = []
    population_history = []
    
    print('Starting genetic algorithm...')
    for i in tqdm(range(num_generations), desc='Progress: '):
        
        fitness = fitness_func(population)
        fitness_history.append(fitness)
        best_individual_history.append(
            get_best_individual(
                population,
                fitness,
                method
            ).tolist()
        )
        population_history.append(population.tolist())

        if verbose:
            tqdm.write(f'Generation {i+1} - Fitness:', end=' ')
            tqdm.write(f'{fitness}')

        parents = selection(fitness, num_parents, population, method)
        offsprings = crossover(parents, num_offsprings, crossover_rate)
        mutants = mutation(offsprings, mutation_rate)

        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

        if check_patience(i, patience, fitness_history):
            break
    
    return population, fitness_history, best_individual_history, population_history

@elapsed_time
def optimize(
    population, 
    num_generations,
    num_parents,
    num_offsprings,
    crossover_rate,
    mutation_rate,
    fitness_func: callable = calc_fitness,
    method: str = 'min',
    patience: int | bool = 200,
    verbose: bool = False,
    filename: str = 'result.json'
) -> tuple:
    """
    Optimizes a population using a genetic algorithm with additional functionalities.

    Args:
        population (numpy.ndarray): The initial population of individuals.
        num_generations (int): The total number of generations.
        num_parents (int): The number of parent individuals.
        num_offsprings (int): The number of offspring individuals.
        crossover_rate (float): The rate of crossover in the genetic algorithm.
        mutation_rate (float): The rate of mutation in the genetic algorithm.
        fitness_func (callable): The fitness function to evaluate individuals.
        method (str): The optimization method, either 'max' or 'min'.
        patience (int | bool): The patience value for early stopping.
        verbose (bool): Whether to display verbose output.
        filename (str): The name of the file to save the optimization results.

    Returns:
        tuple: A tuple containing the optimized parameters and fitness history.
    """

    validate_num_parents_offsprings(population, num_parents, num_offsprings)
    
    patience = validate_patience(patience, num_generations)
    
    population, fitness_history, best_individual_history, population_history = main_optimize(
        population,
        num_generations,
        num_parents,
        num_offsprings,
        crossover_rate,
        mutation_rate,
        fitness_func,
        method,
        patience,
        verbose
    )

    fitness_last_gen = fitness_history[-1]

    if verbose:
        tqdm.write(
            f'\n\nLast generation fitness: {[round(f, 2) for f in fitness_last_gen]}\n'
        )
        tqdm.write(f'Last generation: \n{population}\n') 

    max_or_min_fitness = get_max_or_min_fitness(fitness_last_gen, method)

    parameters = [population[max_or_min_fitness[0][0],:]]
    result = mount_results(
        parameters,
        fitness_history,
        population_history,
        best_individual_history,
        method,
        num_generations,
        num_parents,
        num_offsprings,
        crossover_rate,
        mutation_rate,
        patience,
        population
    )

    save_results(filename, result)

    return parameters, fitness_history