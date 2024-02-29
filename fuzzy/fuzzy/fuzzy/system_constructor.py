from pydantic import BaseModel
from .system import SystemInformation
from .input import Input
from .output import Output


class SystemConstructor:
    
    def __init__(self, fis_file: str):
        self.fis_file = fis_file
        self.system: SystemInformation = None
        self.input: Input = None
        self.output: Output = None


    def _execute(self):
        pass