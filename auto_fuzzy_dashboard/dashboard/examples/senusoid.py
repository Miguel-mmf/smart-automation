from numpy import pi, sin, arange
from pandas import DataFrame


class Senusoid:
    """
    This class creates a sinusoid function.
    """
    
    def __init__(self):
        self._pi = pi
        self.x = [i for i in arange(0, 6 * self._pi, 0.1)]
        self.y = sin(self.x)
        self.values = DataFrame(data={'x':self.x, 'y':self.y}) #.set_index('x',inplace=True)

    @property
    def get_values(self):
        return self.values