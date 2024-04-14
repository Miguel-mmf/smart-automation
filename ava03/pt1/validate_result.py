# %%
import json

from matplotlib import pyplot as plt
from src import calc_gain_circuit
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

# %%
save_fig = True

# %%
values = json.load(open('result.json'))['circuit']

# %%
Vin=values['vin']@u_V
Vs=values['vs']@u_mV
f=values['f']@u_kHz
R1=values['R1']@u_kΩ
R2=values['R2']@u_kΩ
RC=values['RC']@u_kΩ
RE=values['RE']@u_kΩ
C=values['C']@u_uF
CE=values['CE']@u_uF

# %%
circuit = Circuit('Exemplo 1')

circuit.V('input',3, circuit.gnd, Vin)
circuit.AcLine('VS',1,circuit.gnd, rms_voltage=Vs, frequency=f)

circuit.C(1,1,2,C)
circuit.R(2, 2, circuit.gnd, R2)
circuit.R(1, 3, 2, R1)
circuit.R(3,3,4,RC)
circuit.R(4,5,circuit.gnd, RE)
circuit.C(2,5,circuit.gnd, CE)
circuit.BJT(1, 4, 2, 5, model='generic')

circuit.model('generic', 'npn')
print(circuit)

# %%
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.operating_point()
node_values = dict()
for node in analysis.nodes.values():
    node_values[str(node)] = float(node)
    print('Node {}: {:4.1f} V'.format(str(node), float(node)))
i = 0
for node in analysis.nodes.values():
    if i==0:
        fit = float(node)
    i+=1

# %%
gain = calc_gain_circuit(
    Vin=values['vin'],
    Vs=values['vs'],
    f=values['f'],
    R1=values['R1'],
    R2=values['R2'],
    RC=values['RC'],
    RE=values['RE'],
    C=values['C'],
    CE=values['CE'],
    type='DC'
)

# %%
print(f'\nO ganho DC do circuito é: {gain}')

# %%
aux = simulator.transient(step_time=1@u_us, end_time=10@u_ms)

# %%
entrada = [float(val) for val in aux['1']]
saida = [float(val) for val in aux['5']]

# %%
print(f'Ganho AC do amplificador: {round((max(saida) - min(saida))/(max(entrada) - min(entrada)), 2)}')

# %%
figure, axs = plt.subplots(2,1, figsize=(12, 6), sharex=True)
axs[0].plot(aux.time, aux['1'], label='Tensão no ponto 1', linewidth=3)
axs[1].plot(aux.time, aux['2'], label='Tensão no ponto 2', linewidth=3)

axs[0].set_ylabel('Tensão (V)')
axs[1].set_ylabel('Tensão (V)')
axs[1].set_xlabel('Tempo (s)')
axs[0].legend()
axs[1].legend()
axs[0].grid()
axs[1].grid()

plt.tight_layout()
if save_fig:
    plt.savefig(
        './images/tensoes_12_resultado_AG.png',
        dpi=300,
    )
    plt.savefig(
        './images/tensoes_12_resultado_AG.pdf',
        dpi=300,
    )
else:
    plt.show()

# %%
figure, axs = plt.subplots(2,1, figsize=(12, 4), sharex=True)
axs[0].plot(aux.time, aux['4'], label='Tensão no ponto 4', linewidth=3)
axs[1].plot(aux.time, aux['5'], label='Tensão no ponto 5', linewidth=3)

axs[0].set_ylabel('Tensão (V)')
axs[1].set_ylabel('Tensão (V)')
axs[1].set_xlabel('Tempo (s)')
axs[0].legend()
axs[1].legend()
axs[0].grid()
axs[1].grid()

plt.tight_layout()
if save_fig:
    plt.savefig(
        './images/tensoes_45_resultado_AG.png',
        dpi=300,
    )
    plt.savefig(
        './images/tensoes_45_resultado_AG.pdf',
        dpi=300,
    )
else:
    plt.show()

# %%
figure, axs = plt.subplots(2,1, figsize=(12, 4), sharex=True)
axs[0].plot(aux.time, aux['5'], label='Tensão no ponto 5', linewidth=3, color='blue')
axs[1].plot(aux.time, aux['1'], label='Tensão no ponto 1', linewidth=3, color='red')

axs[0].set_ylabel('Tensão (V)')
axs[1].set_ylabel('Tensão (V)')
axs[1].set_xlabel('Tempo (s)')
axs[0].legend()
axs[1].legend()
axs[0].grid()
axs[1].grid()

plt.tight_layout()
if save_fig:
    plt.savefig(
        './images/tensoes_15_resultado_AG.png',
        dpi=300,
    )
    plt.savefig(
        './images/tensoes_15_resultado_AG.pdf',
        dpi=300,
    )
else:
    plt.show()