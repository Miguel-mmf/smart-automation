from simpful import FuzzySystem
# from .variable import Variable


class FuzzySystemRegressor(FuzzySystem):
    
    def __init__(self, linguistic_vars: list[tuple], method: str = "Mamdani"):
        """
        This class is responsible for creating the fuzzy system.

        Args:
            - name: Name of the fuzzy system
            - rules: The set of rules
            - weights: The set of rules' weights
        """
        super().__init__()
        self._variables = linguistic_vars
        self._method = method
        
        for var_name, var in self._variables:
            self.add_linguistic_variable(var_name, var)