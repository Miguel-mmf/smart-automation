import numpy as np
from .binary_to_decimal import binary_to_decimal


def fitness_function(value):
    return (2000 - value)**2


def calc_fitness(
    pop: np.array,
    Vth: int | float = 10,
    Rl: int | float = 10,
    Rth: int | float = 50,
    fitness_function: callable = fitness_function,
    verbose: bool = False
) -> list:
    
    b_values = [''.join(map(str, pop[i,:])) for i in range(pop.shape[0])]
    d_values = [binary_to_decimal(b) for b in b_values]
    
    if verbose:
        print(f'Binary values: {b_values}')
        print(f'Decimal values: {d_values}')
    
    return [fitness_function(d, Rth, Vth) for d in d_values]