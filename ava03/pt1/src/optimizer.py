import numpy as np
from tqdm import tqdm

from .binary_to_decimal import binary_to_decimal
from .core import crossover, mutation, selection
from .fitness import calc_fitness


def optimize(
    population, 
    num_generations,
    num_parents,
    num_offsprings,
    crossover_rate,
    mutation_rate,
    fitness_func: callable = calc_fitness,
    max: bool = True,
    min: bool = False,
    verbose: bool = False
) -> tuple:
    
    parameters, fitness_history = [], []
    print('Starting genetic algorithm...')
    for i in tqdm(range(num_generations), desc='Progress: '):
        tqdm.write(f'Generation {i+1} - Fitness:', end=' ')
        fitness = fitness_func(population)
        fitness_history.append(fitness)
        tqdm.write(f'{fitness}\n')
        
        parents = selection(fitness, num_parents, population)
        
        offsprings = crossover(parents, num_offsprings, crossover_rate)
        
        mutants = mutation(offsprings, mutation_rate)
        
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants

    
    fitness_last_gen = fitness_func(population)
    
    if verbose:
        tqdm.write('Fitness da última geração: {}\n'.format([round(f,2) for f in fitness_last_gen]))
        tqdm.write('Última geração: \n{}\n'.format(population)) 
        # print('Última geração: \n{}\n'.format(binary_to_decimal(population)))
    
    if max:
        max_or_min_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    elif min:
        max_or_min_fitness = np.where(fitness_last_gen == np.min(fitness_last_gen))
    elif not max and not min:
        raise ValueError('You must set max=True or min=True')
    elif max and min:
        raise ValueError('You must set only one of max=True or min=True')
    else:
        raise ValueError('You must set max=True or min=True')
    
    parameters.append(population[max_or_min_fitness[0][0],:])
    
    return parameters, fitness_history