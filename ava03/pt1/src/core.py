import numpy as np
import random as rd
from random import randint, random


def random_value_with_constraint(
    index: int,
):
    val = random()
    if index == 0:
        return val*100 if val != 0 else 45
    elif index in [1, 2]:
        return val*10 if val != 0 else 1
    elif index == 3:
        return val*3 if val != 0 else 1
    else:
        return val*20 if val != 0 else 1


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
    # initial_population = np.random.randint(1, 1000, size = pop_size)
    initial_population = np.zeros(shape=pop_size, dtype=float)
    for row in range(num_ind):
        initial_population[row, :] = [
            random_value_with_constraint(0),
            random_value_with_constraint(1),
            random_value_with_constraint(2),
            random_value_with_constraint(3),
            random_value_with_constraint(4),
            random_value_with_constraint(5)
        ]
    
    if verbose:
        print('Tamanho da População = {}'.format(pop_size))
        print('População Inicial: \n{}'.format(initial_population))
    
    return initial_population


def selection(
    fitness: list,
    num_parents: int,
    population: np.array,
    method: str = 'min'
) -> tuple:
    """This function selects the parents for the crossover.

    Args:
        fitness (list): The fitness of each individual in the population.
        num_parents (int): The number of parents to be selected.
        population (np.array): The population.
    
    Returns:
        tuple: The parents selected for the crossover.
    """
    
    # fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    
    for i in range(num_parents):
        if method == 'max':
            fitness_idx = np.where(fitness == np.max(fitness))
        elif method == 'min':
            fitness_idx = np.where(fitness == np.min(fitness))
        else:
            raise ValueError('Método inválido. Escolha entre "max" ou "min"')
        
        # fitness_idx = np.where(fitness == np.min(fitness))
        parents[i,:] = population[fitness_idx[0][0], :]
        fitness[fitness_idx[0][0]] = -999999 if method == 'max' else 999999
    
    return parents


def crossover(
    parents,
    num_offsprings,
    crossover_rate
) -> np.array:
    """_summary_

    Args:
        parents (_type_): _description_
        num_offsprings (_type_): _description_
        crossover_rate (_type_): _description_
    
    Docs:
        Para criar um par de novos indivíduos, dois pais geralmente são escolhidos a partir do atual geração, e partes de seus cromossomos são trocados (cruzados) para criar dois novos cromossomos representando a prole. Essa operação é chamada de cruzamento, ou recombinação.

    Returns:
        np.array: _description_
    """    
    
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
            random_sum_diff = rd.random()
            if random_sum_diff < 0.5:
                mutants[i,int_random_value] = random_value_with_constraint(int_random_value)
            else:
                mutants[i,int_random_value] = random_value_with_constraint(int_random_value)
    
    return mutants


if __name__ == '__main__':
    
    initial_pop(
        num_ind=10,
        num_elements=6,
        verbose=True
    )