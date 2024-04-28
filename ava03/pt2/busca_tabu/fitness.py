import numpy as np

def calc_fitness(weight, value, population, threshold) -> tuple:
    """
    Calculates the fitness of individuals in a population based on weight, value, and a threshold.

    Args:
        weight (numpy.ndarray): The weight values for each item.
        value (numpy.ndarray): The value associated with each item.
        population (numpy.ndarray): The population matrix where each row represents an individual.
        threshold (float): The maximum weight threshold an individual can have.

    Returns:
        numpy.ndarray: An array of fitness scores for each individual in the population based on the threshold.
    """

    total_weight = np.dot(population, weight)
    total_value = np.dot(population, value)
    return np.where(total_weight <= threshold, total_value, 0), total_weight, total_value