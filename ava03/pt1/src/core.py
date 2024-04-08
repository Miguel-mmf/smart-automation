import numpy as np
import random as rd
from random import randint

def initial_pop(
    num_ind: int,
    num_elements: int = 6,
    verbose: bool = False
) -> tuple:
    """This function generates the initial population of the genetic algorithm.

    Args:
        num_ind (int): The number of individuals in the population.
        bits (_type_): The number of bits used to represent each individual.
        xmin (_type_): The constraint of the minimum value of the individual.
        xmax (_type_): The constraint of the maximum value of the individual.

    Returns:
        tuple: The initial population and the binary representation of the initial population.
    """
    
    # def validate_initial_pop(pop):
        
    #     b_values = [''.join(map(str, pop[i,:])) for i in range(pop.shape[0])]
    #     d_values = [binary_to_decimal(b) for b in b_values]
    #     if 0 in d_values:
    #         return [index for index, value in enumerate(d_values) if value == 0]
    #     return []
    
    if not num_elements:
        # Sequencia
            # R1=values[0],
            # R2=values[1],
            # RC=values[2],
            # RE=values[3],
            # C=values[4],
            # CE=values[5]
        num_elements = 6
    
    
    pop_size = (num_ind, num_elements)
    initial_population = np.random.randint(100, size = pop_size)
    
    # invalid_pop = validate_initial_pop(initial_population)
    # while invalid_pop:
    #     initial_population[invalid_pop] = np.random.randint(2, size = (len(invalid_pop), bits))
    #     invalid_pop = validate_initial_pop(initial_population)
    
    if verbose:
        print('Tamanho da População = {}'.format(pop_size))
        print('População Inicial: \n{}'.format(initial_population))
    
    return initial_population


def selection(
    fitness: list,
    num_parents: int,
    population: np.array,
) -> tuple:
    """This function selects the parents for the crossover.

    Args:
        fitness (list): The fitness of each individual in the population.
        num_parents (int): The number of parents to be selected.
        population (np.array): The population.
    
    Returns:
        tuple: The parents selected for the crossover.
    """
    
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
        
    return parents


def crossover(
    parents,
    num_offsprings,
    crossover_rate
) -> np.array:
    
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    i=0
    while (i < num_offsprings):
        x = rd.random()
        parent1_index = 0
        parent2_index = 1
        if x < crossover_rate:
            offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
            offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
            offsprings[i+1,0:crossover_point] = parents[parent2_index,0:crossover_point]
            offsprings[i+1,crossover_point:] = parents[parent1_index,crossover_point:]
        else:
            offsprings[i] = parents[parent1_index]
            offsprings[i+1] = parents[parent2_index]
        i+=2 
        
    return offsprings


def mutation(
    offsprings,
    mutation_rate
) -> np.array:
    
    mutants = np.empty((offsprings.shape))
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i,:] = offsprings[i,:]
        if random_value < mutation_rate:
            int_random_value = randint(0,offsprings.shape[1]-1)    
            if mutants[i,int_random_value] == 0 :
                mutants[i,int_random_value] = 1
            else :
                mutants[i,int_random_value] = 0
    return mutants   