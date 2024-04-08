# %%
from src import (
    initial_pop,
    calc_fitness,
    optimize,
    create_plot
)
# %%
init_pop = initial_pop(
    num_ind=10,
    num_elements=6,
    verbose=True
)
# %%
calc_fitness(
    init_pop,
    verbose=True
)
# %%
parameters, fitness_history = optimize(
    population=init_pop,
    num_generations=100,
    num_parents=4,
    num_offsprings=6,
    crossover_rate=0.7,
    mutation_rate=0.3,
    max=False,
    min=True,
    verbose=True
)
# %%
print(f'Os itens selecionados que resultam no maior valor são: {list(parameters[0])}\n')
# %%
create_plot(
    fitness_history=fitness_history,
    num_generations=100,
    save=True
)
# %%
