import json
import humps
import numpy as np
from tqdm import tqdm
from time import time
from .binary_to_decimal import binary_to_decimal
from .core import crossover, mutation, selection
from .fitness import calc_fitness

def mount_results(parameters):
    
    result = dict()
    result['Vs'] = 1
    result['Vin'] = 22
    result['f'] = 1
    result['R1'] = round(parameters[0][0], 2)
    result['R2'] = round(parameters[0][1], 2)
    result['RC'] = round(parameters[0][2], 2)
    result['RE'] = round(parameters[0][3], 2)
    result['C'] = round(parameters[0][4], 2)
    result['CE'] = round(parameters[0][5], 2)
    
    return result

def get_best_individual(population, fitness, method):
    if method == 'max':
        fitness_idx = np.where(fitness == np.max(fitness))
    elif method == 'min':
        fitness_idx = np.where(fitness == np.min(fitness))

    return population[fitness_idx[0][0], :]


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
    
    if num_offsprings + num_parents != population.shape[0]:
        raise ValueError('The number of offsprings add and the number of parents must be equal to the population size.')
    
    if not patience:
        patience = num_generations
    
    start = time()
    parameters = list()
    fitness_history = list()
    best_individual_history = list()
    population_history = list()
    result = dict()
    
    print('Starting genetic algorithm...')
    for i in tqdm(range(num_generations), desc='Progress: '):
        if verbose:
            tqdm.write(f'Generation {i+1} - Fitness:', end=' ')
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
            tqdm.write(f'{fitness}')
        
        parents = selection(fitness, num_parents, population, method)
        
        offsprings = crossover(parents, num_offsprings, crossover_rate)
        
        mutants = mutation(offsprings, mutation_rate)
        
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        
        if i > patience:
            if np.all(fitness_history[-patience] == fitness_history[-1]):
                tqdm.write(f'Early stopping at generation {i+1}')
                break

    
    fitness_last_gen = fitness_func(population)
    
    if verbose:
        tqdm.write('\n\nLast generation fitness: {}\n'.format([round(f,2) for f in fitness_last_gen]))
        tqdm.write('Last generation: \n{}\n'.format(population)) 
    
    if method == 'max':
        max_or_min_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    elif method == 'min':
        max_or_min_fitness = np.where(fitness_last_gen == np.min(fitness_last_gen))
    else:
        raise ValueError('Método inválido. Escolha entre "max" ou "min"')
    
    parameters.append(population[max_or_min_fitness[0][0],:])
    
    result['circuit'] = mount_results(parameters)
    result['fitness_last_generation'] = [round(f,2) for f in fitness_last_gen]
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
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f'Time elapsed: {round(time() - start, 2)}s')
    return parameters, fitness_history