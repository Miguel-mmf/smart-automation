import numpy as np

from .core import crossover, mutation, selection
from .fitness import calc_fitness


def optimize(
    population, 
    num_generations,
    num_parents,
    num_offsprings,
    crossover_rate,
    mutation_rate,
    verbose: bool = False
) -> tuple:
    
    parameters, fitness_history = [], []
    for i in range(num_generations):
        fitness = calc_fitness(population)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings, crossover_rate)
        mutants = mutation(offsprings, mutation_rate)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
    
    if verbose:
        print('Última geração: \n{}\n'.format(population)) 
    fitness_last_gen = calc_fitness(population)
    if verbose:
        print('Fitness da última geração: {}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0],:])
    
    return parameters, fitness_history