import numpy as np
from .create_circuit import calc_gain_circuit


def fitness_function_p1(values):
    """This function calculates the fitness of the individual.
    
    Circuit: https://github.com/Miguel-mmf/smart-automation/blob/ea604ba3c2b27e4210303b60d4108f707e1c465a/ava03/pt1/circuito1.png

    Args:
        values (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    gain = calc_gain_circuit(
        Vin=22,
        Vs=1,
        f=1,
        R1=values[0],
        R2=values[1],
        RC=values[2],
        RE=values[3],
        C=values[4],
        CE=values[5],
        type='DC'
    )
    
    return round((2000 - gain)**2, 2) # **2


def calc_fitness(
    pop: np.array,
    fitness_function: callable = fitness_function_p1,
    verbose: bool = False
) -> list:
    
    return [
        fitness_function(list(pop[i,:]))
        for i in range(pop.shape[0])
    ]