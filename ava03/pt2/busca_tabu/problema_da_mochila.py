# %%
import numpy as np
import pandas as pd
from plots import create_plot
from core import (
    # binary_to_decimal,
    initial_solutions,
    tabu_search
)
from datetime import date
from fitness import calc_fitness

# %%
# Problema da Mochila
    # Um ladrão entra em uma loja carregando mochila (bolsa) que
    # pode transportar 35 kg de peso. A loja possui 10 itens, cada
    # um com um peso e preço específicos.
    # Agora, o dilema do ladrão é fazer uma seleção de itens que
    # maximize o valor (ou seja, preço total) sem exceder o peso da
    # mochila. Temos que ajudar o ladrão a fazer a seleção.
item_number = np.arange(1,11)
weight = np.array([10,8,2,4,15,5,3,1,12,9]) # weight = np.random.randint(1, 15, size = 10)
value = np.array([1200,200,300,1000,1500,800,2000,40,500,3000]) # value = np.random.randint(10,3000, size = 10)

# %%
print('Lista de itens da casa com seus respectivos valores e pesos:')
print('Item No.   Peso   Valor')
for i in range(item_number.shape[0]):
    print(f'{item_number[i]}          {weight[i]}         {value[i]}')

# %%
population = initial_solutions(weight.shape[0], True)
# %%
fitness = calc_fitness(weight, value, population, 35)
# %%
# best_solution = population[np.argmax(fitness)]
# best_solution_fitness = np.max(fitness)
# %%
filename = f'./tabu_search_results_{date.today()}.json'
patience = 40
# %%
(
    best_solution,
    best_solution_fitness,
    solution_history,
    solution_fitness_history
) = tabu_search(
    population=population,
    num_iterations=1000,
    weight=weight,
    value=value,
    threshold=35,
    tabu_list_size=2,
    method='max',
    patience=30,
    filename=filename,
    verbose=False
)

# print(f'Melhor Indivíduo: {best_solution}')
# print(f'Fitness do Melhor Indivíduo: {best_solution_fitness}')

# %%
create_plot(
    solution_fitness_history,
    save=True,
    method='max',
)
# %%
print(f'Melhor Indivíduo: {best_solution}')
print(f'Fitness do Melhor Indivíduo: {best_solution_fitness}')
# %%