# %%
import numpy as np
import pandas as pd
from plots import create_plot
from core import (
    # binary_to_decimal,
    initial_solutions,
    tabu_search
)
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
data = pd.DataFrame({'Item No.': item_number, 'Peso': weight, 'Valor': value})
data.to_csv('mochila.csv', index=False)
print('Lista de itens da casa com seus respectivos valores e pesos:')
print('Item No.   Peso   Valor')
for i in range(item_number.shape[0]):
    print(f'{item_number[i]}          {weight[i]}         {value[i]}')

# %%
pop = initial_solutions(item_number.shape[0], True)
# %%
fitness = calc_fitness(weight, value, pop, 35)
# %%
best_solution = pop[np.argmax(fitness)]
best_solution_fitness = np.max(fitness)
# %%
(
    best_solution,
    best_solution_fitness,
    solution_history,
    solution_fitness_history
) = tabu_search(
    # population=pop,
    population=initial_solutions(item_number.shape[0], True),
    num_iterations=100,
    weight=weight,
    value=value,
    threshold=35,
    tabu_list_size=5,
    method='max',
    patience=20
)

print(f'Melhor Indivíduo: {best_solution}')
print(f'Fitness do Melhor Indivíduo: {best_solution_fitness}')

# %%
create_plot(
    solution_fitness_history,
    save=True,
    method='max',
)

# %%
# iteration = 0
# best_iteration = 0
# tabu_list = []
# solution_history = []
# solution_fitness_history = []
# # %%
# while (iteration - best_iteration) < 10:
#     iteration += 1
#     print(f'Iteração {iteration}')
#     print(f'Melhor Indivíduo: {best_solution}')
#     print(f'Fitness do Melhor Indivíduo: {best_solution_fitness}')
#     print(f'Lista Tabu: {tabu_list}')
    
#     # Gera os vizinhos
#     neighbors = []
#     for pos in range(10):
#         neighbor = best_solution.copy()
#         neighbor[pos] = 1 - neighbor[pos]
#         neighbors.append(
#             neighbor
#             if binary_to_decimal(''.join(map(str, neighbor))) not in tabu_list # caso ele esteja na lista tabu, esse max ja e conhecido
#             else np.ones(10) # isso faz com que ele nao seja factível
#         )
    
#     solution_history.append(neighbors)

#     # Calcula o fitness dos vizinhos
#     fitness_neighbors = calc_fitness(weight, value, np.array(neighbors), 35)
#     solution_fitness_history.append(fitness_neighbors)
    
#     # Seleciona o melhor vizinho
#     best_neighbor = neighbors[np.argmax(fitness_neighbors)]
#     best_neighbor_fitness = np.max(fitness_neighbors)

#     # Adiciona o melhor vizinho à lista tabu
#     tabu_list.append(binary_to_decimal(''.join(map(str, best_neighbor))))
#     # Remove o pior indivíduo da lista tabu
#     if len(tabu_list) > 5:
#         tabu_list.pop(0)

#     # Atualiza o melhor indivíduo
#     if best_neighbor_fitness > best_solution_fitness:
#         best_solution = best_neighbor
#         best_solution_fitness = best_neighbor_fitness
#         best_iteration = iteration

# %%
print(f'Melhor Indivíduo: {best_solution}')
print(f'Fitness do Melhor Indivíduo: {best_solution_fitness}')
# %%