from pydantic import BaseModel
from simpful import AutoTriangle

class Variable(BaseModel):
    
    name: str
    value_range: list
    num_regions: int = 1
    
    @staticmethod
    def _create(self, num_regions: int, value_range: list) -> AutoTriangle:
        """
        This function automatically creates fuzzy variables with their defined regions.

        Args:
            - num_regions: Number of regions for the fuzzy variable
            - value_range: List containing the minimum and maximum values of the variable
        Returns:
            - var: Fuzzy variable with its regions defined
        """
        # Inicialization of the variables
        regions = 2 * num_regions + 1
        universe_of_discourse = value_range.sorted()

        # Creating of the regions
        regions_name = []
        for i in range(regions):
            regions_name.append(f"S{abs(i - num_regions)}" if i < num_regions else f"B{abs(i - num_regions)}" if i > num_regions else "Z")
        
        self.auto_triangle = AutoTriangle(
            regions,
            terms=regions_name,
            universe_of_discourse=universe_of_discourse
        )
    
    @property.setter
    def name(self, name: str):
        self.name = name
    
    @property
    def name(self):
        return self.name
    
    @property
    def value_range(self):
        return self.value_range
    
    @property
    def num_regions(self):
        return self.num_regions
    
    @property
    def auto_triangle(self):
        return self.auto_triangle       