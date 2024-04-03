from numpy import random
from max_transfer_potencia.src.binary_to_decimal import binary_to_decimal


def inital_pop(
    L: int,
    bits: int,
    xmin: int | float = 0,
    xmax: int | float = 1
) -> tuple:
    """This function generates the initial population of the genetic algorithm.

    Args:
        L (int): The number of individuals in the population.
        bits (_type_): The number of bits used to represent each individual.
        xmin (_type_): The constraint of the minimum value of the individual.
        xmax (_type_): The constraint of the maximum value of the individual.

    Returns:
        tuple: The initial population and the binary representation of the initial population.
    """
    
    temp = round(random.rand(L,bits),2)    
    Po_Binario = int(temp) #.astype('int')
    individuos = [
        int(binary_to_decimal(Po_Binario[i,:]))
        for i in range(0,L)
    ]
    Po = [
        xmin + individuos[i]*(xmax-xmin)/(2**bits - 1)
        for i in range(0,L)
    ]

    return Po, Po_Binario


def fitness_avaliator(
    Po: list,
    bits: int,
    L: int,
    xmin: int | float = 0,
    xmax: int | float = 1
) -> list:
    """This function evaluates the fitness of the population.

    Args:
        Po (list): The population.
        bits (_type_): The number of bits used to represent each individual.
        L (int): The number of individuals in the population.
        xmin (_type_): The constraint of the minimum value of the individual.
        xmax (_type_): The constraint of the maximum value of the individual.

    Returns:
        list: The fitness of each individual in the population.
    """

    fitness = [
        Po[i] * (Po[i] - 1)
        for i in range(0,L)
    ]

    return fitness


def selection(
    Po: list,
    Po_Binario: list,
    fitness: list,
    bits: int,
    L: int,
    xmin: int | float = 0,
    xmax: int | float = 1
) -> tuple:
    """This function selects the parents for the crossover.

    Args:
        Po (list): The population.
        Po_Binario (list): The binary representation of the population.
        fitness (list): The fitness of each individual in the population.
        bits (_type_): The number of bits used to represent each individual.
        L (int): The number of individuals in the population.
        xmin (_type_): The constraint of the minimum value of the individual.
        xmax (_type_): The constraint of the maximum value of the individual.

    Returns:
        tuple: The parents selected for the crossover.
    """

    # Normalization of the fitness
    fitness = [
        fitness[i] / sum(fitness)
        for i in range(0,L)
    ]

    # Cumulative sum of the normalized fitness
    cumsum = [sum(fitness[0:i+1]) for i in range(0,L)]

    # Selection of the parents
    parents = []
    for i in range(0,L):
        r = random.rand()
        for j in range(0,L):
            if r < cumsum[j]:
                parents.append(j)
                break

    # Parents selected for the crossover
    Po_Pais = [
        Po[parents[i]]
        for i in range(0,L)
    ]
    Po_Binario_Pais = [
        Po_Binario[parents[i]]
        for i in range(0,L)
    ]

    return Po_Pais, Po_Binario_Pais