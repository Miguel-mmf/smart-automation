import numpy as np
from .binary_to_decimal import binary_to_decimal
from .create_circuit import calc_gain_circuit


def fitness_function(values):
    
    gain = calc_gain_circuit(
        Vin=22,
        Vs=0.1,
        f=1e3,
        R1=values[0],
        R2=values[1],
        RC=values[2],
        RE=values[3],
        C=values[4],
        CE=values[5]
    )
    
    return round((2000 - gain)**2, 2)


def calc_fitness(
    pop: np.array,
    fitness_function: callable = fitness_function,
    verbose: bool = False
) -> list:
    
    return [
        fitness_function(list(pop[i,:]))
        for i in range(pop.shape[0])
    ]