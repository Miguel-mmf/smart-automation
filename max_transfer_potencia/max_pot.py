# %%
from src import (
    initial_pop,
    calc_fitness,
    optimize,
    create_plot
)
# %%
Vth = 10
Rl = 10
Rth = 50
print(f'Pmax = {Vth**2 / (4*Rth)}')
# %%
init_pop = initial_pop(
    num_ind=10,
    bits=8,
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
    num_generations=10,
    save=True
)
# %%
