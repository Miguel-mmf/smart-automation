# %%
from src import initial_pop, optimize
from src import calc_fitness
from src.plots import create_plot
import matplotlib.pyplot as plt
# %%
init_pop = initial_pop(
    num_ind=10,
    bits=4,
    verbose=True
)
# %%
calc_fitness(init_pop)
# %%
parameters, fitness_history = optimize(
    population=init_pop,
    num_generations=10,
    num_parents=4,
    num_offsprings=6,
    crossover_rate=0.8,
    mutation_rate=0.1,
    verbose=True
)
# %%
print(f'Os itens selecionados que resultam no maior valor são: {list(parameters[0])}\n')
# %%
create_plot(
    fitness_history=fitness_history,
    num_generations=10
)
# %%
