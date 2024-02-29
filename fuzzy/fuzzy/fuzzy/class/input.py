from pydantic import BaseModel
from typing import List


class Input(BaseModel):
    Name: str
    Range: list
    NumMFs: int
    mfs: List[dict]
        
    @classmethod
    def _init_mfs(cls, mfs: List[dict]):
        for i, mf in enumerate(mfs):
            setattr(cls, f'MF{i+1}', mf)