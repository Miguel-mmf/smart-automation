import numpy as np
from .binary_to_decimal import binary_to_decimal


def calc_fitness(pop):
    Vth = 10
    Rl = 10
    Rth = 50
    
    def fitness_function(value):
        return Vth**2 * value / ((Rth * value)**2)
    
    b_values = [''.join(map(str, pop[i,:])) for i in range(pop.shape[0])]
    d_values = [binary_to_decimal(b) for b in b_values]
    print(f'Binary values: {b_values}')
    print(f'Decimal values: {d_values}')
    return [fitness_function(d) for d in d_values]


# def cal_fitness(weight, value, population, threshold):
#     fitness = np.empty(population.shape[0])
#     for i in range(population.shape[0]):
#         S1 = np.sum(population[i] * value)
#         S2 = np.sum(population[i] * weight)
#         if S2 <= threshold:
#             fitness[i] = S1
#         else :
#             fitness[i] = 0 
#     return fitness


# def fitness_avaliator(
#     Po: list,
#     bits: int,
#     L: int,
#     xmin: int | float = 0,
#     xmax: int | float = 1
# ) -> list:
#     """This function evaluates the fitness of the population.

#     Args:
#         Po (list): The population.
#         bits (_type_): The number of bits used to represent each individual.
#         L (int): The number of individuals in the population.
#         xmin (_type_): The constraint of the minimum value of the individual.
#         xmax (_type_): The constraint of the maximum value of the individual.

#     Returns:
#         list: The fitness of each individual in the population.
#     """

#     fitness = [
#         Po[i] * (Po[i] - 1)
#         for i in range(0,L)
#     ]

#     return fitness