def fitness_func(
    Vth: float,
    # Vdd: float,
    R: float,
    Rth: float,
) -> float:
    """This function evaluates the fitness of the population.

    Args:
        Vth (float): The threshold voltage.
        Vdd (float): The drain voltage.
        R (float): The resistance.
        Rth (float): The thermal resistance.

    Returns:
        float: The fitness of the individual.
    """

    return (Vth**2)*R / ((R + Rth)**2)