from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

import warnings
warnings.filterwarnings("ignore")


def calc_gain_circuit(
    Vin: float | int,
    Vs: float | int,
    f: float | int,
    R1: float | int,
    R2: float | int,
    RC: float | int,
    RE: float | int,
    C: float | int,
    CE: float | int = 20,
    type: str = 'DC',
    return_Is_current: bool = False
):
    """This function creates a circuit with the given parameters.
    
    The circuit is a common emitter amplifier with a bypass capacitor.
    
    Circuit: https://github.com/Miguel-mmf/smart-automation/blob/main/ava03/pt1/circuito1.png

    Args:
        Vin (float | int): The input DC voltage.
        Vs (float | int): The input AC voltage.
        f (float | int): The frequency of the input AC voltage.
        R1 (float | int): The resistance of the first resistor.
        R2 (float | int): The resistance of the second resistor.
        RC (float | int): The resistance of the collector resistor.
        RE (float | int): The resistance of the emitter resistor.
        C (float | int): The capacitance of the capacitor.
        CE (float | int, optional): The capacitance of the emitter capacitor. Bypass capacitor. Defaults to 20.
    """
    
    def calc_AC_gain(analysis):
        """This function calculates the AC gain of the circuit.

        Args:
            analysis (_type_): The analysis of the circuit.

        Returns:
            float: The gain of the circuit.
        """
        entrada = [float(val) for val in analysis['1']]
        saida = [float(val) for val in analysis['5']]

        return round((max(saida) - min(saida))/(max(entrada) - min(entrada)), 4)
    
    
    def calc_DC_gain(analysis, Vs=Vs):
        """This function calculates the DC gain of the circuit.

        Args:
            analysis (_type_): The analysis of the circuit.

        Returns:
            float: The gain of the circuit.
        """
        return round(float(analysis.nodes['5'])/float(Vs/1000), 0)

    def calc_Is(analysis, R1=R1, RC=RC):
        """This function calculates the Is current of the circuit.

        Args:
            analysis (_type_): The analysis of the circuit.

        Returns:
            float: The Is current of the circuit.
        """
        
        Is = (float(analysis.nodes['3']) - float(analysis.nodes['4'])) / float(R1)
        Is += (float(analysis.nodes['3']) - float(analysis.nodes['4'])) / float(RC)
        
        return Is
    
    
    Vin = Vin@u_V
    Vs = Vs@u_mV
    f = f@u_kHz
    R1 = R1@u_kΩ
    R2 = R2@u_kΩ
    RC = RC@u_kΩ
    RE = RE@u_kΩ
    C = C@u_uF
    CE = CE@u_uF
    
    circuit = Circuit('Circuit')

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
    
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    if not return_Is_current:
        if type == 'DC':
            analysis = simulator.operating_point()
            return calc_DC_gain(analysis)
        
        elif type == 'AC':
            analysis = simulator.transient(
                step_time=0.5@u_us,
                end_time=1@u_ms
            )
            return calc_AC_gain(analysis)
        else:
            raise ValueError('Invalid type. Choose between "DC" and "AC".')
    else:
        
        
        
        if type == 'DC':
            analysis = simulator.operating_point()
            return (calc_DC_gain(analysis), calc_Is(analysis))
        
        elif type == 'AC':
            analysis = simulator.transient(
                step_time=0.5@u_us,
                end_time=1@u_ms
            )
            return (calc_AC_gain(analysis), calc_Is(analysis))