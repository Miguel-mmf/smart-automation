from pydantic import BaseModel


class SystemInformation(BaseModel):
    Name: str
    Type: str
    Version: float
    NumInputs: int
    NumOutputs: int
    NumRules: int
    AndMethod: str
    OrMethod: str
    ImpMethod: str
    AggMethod: str
    DefuzzMethod: str
    
    def __str__(self):
        return f"System Information\n" \
               f"Name: {self.Name}\n" \
               f"Type: {self.Type}\n" \
               f"Version: {self.Version}\n" \
               f"NumInputs: {self.NumInputs}\n" \
               f"NumOutputs: {self.NumOutputs}\n" \
               f"NumRules: {self.NumRules}\n" \
               f"AndMethod: {self.AndMethod}\n" \
               f"OrMethod: {self.OrMethod}\n" \
               f"ImpMethod: {self.ImpMethod}\n" \
               f"AggMethod: {self.AggMethod}\n" \
               f"DefuzzMethod: {self.DefuzzMethod}\n"