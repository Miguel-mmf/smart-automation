# %%
from functools import partial
from src import (
    initial_pop,
    calc_fitness,
    optimize,
    create_plot,
    calc_gain_circuit
)
# %%
problem = 'p2'
num_ind = 15
num_elements = 6
num_generations = 1000
num_parents = 9
num_offsprings = 6
crossover_rate = 0.4
mutation_rate = 0.08
method = 'min'
patience = 300
verbose = True
filename = 'result_p2.json'
# %%
init_pop = initial_pop(
    num_ind=num_ind,
    num_elements=num_elements,
    verbose=verbose
)

# %%
def fitness_function_p2(values):

    gain, Is = calc_gain_circuit(
        Vin=22,
        Vs=1,
        f=1,
        R1=values[0],
        R2=values[1],
        RC=values[2],
        RE=values[3],
        C=values[4],
        CE=values[5],
        type='DC',
        return_Is_current=True
    )

    return round((2000 - gain)**2 + 1*Is, 2)

# %%
calc_fitness(
    init_pop,
    fitness_function=fitness_function_p2,
    verbose=True
)

# %%
parameters, fitness_history = optimize(
    population=init_pop,
    num_generations=num_generations,
    num_parents=num_parents,
    num_offsprings=num_offsprings,
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate,
    fitness_func=partial(calc_fitness, fitness_function=fitness_function_p2),
    method=method,
    patience=patience,
    verbose=verbose,
    filename=filename
)
# %%
print(f'Os itens selecionados que resultam no maior valor são: {list(parameters[0])}\n')
# %%
# create_plot(
#     fitness_history=fitness_history,
#     # num_generations=num_generations if not patience else len(fitness_history),
#     method=method,
#     save=True,
#     problem=problem
# )
# %%
# import os

# os.system('python validate_result.py')