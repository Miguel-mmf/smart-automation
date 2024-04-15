import numpy as np
import pandas as pd
from simpful import FuzzySystem, FuzzySet, LinguisticVariable, Triangular_MF, Trapezoidal_MF


def get_rules() -> list:
    R1 = "IF (Erro IS EGN) AND (derivadaErro IS DEGN) THEN (deltaFrequencia IS DFN)"
    R2 = "IF (Erro IS EGN) AND (derivadaErro IS DEN) THEN (deltaFrequencia IS DGFN)"
    R3 = "IF (Erro IS EGN) AND (derivadaErro IS DEZ) THEN (deltaFrequencia IS DGFN)"
    R4 = "IF (Erro IS EGN) AND (derivadaErro IS DEP) THEN (deltaFrequencia IS DGFN)"
    R5 = "IF (Erro IS EGN) AND (derivadaErro IS DEGP) THEN (deltaFrequencia IS DGFN)"
    R6 = "IF (Erro IS EN) AND (derivadaErro IS DEGN) THEN (deltaFrequencia IS DFN)"
    R7 = "IF (Erro IS EN) AND (derivadaErro IS DEN) THEN (deltaFrequencia IS DFN)"
    R8 = "IF (Erro IS EN) AND (derivadaErro IS DEZ) THEN (deltaFrequencia IS DFN)"
    R9 = "IF (Erro IS EN) AND (derivadaErro IS DEP) THEN (deltaFrequencia IS DFN)"
    R10 = "IF (Erro IS EN) AND (derivadaErro IS DEGP) THEN (deltaFrequencia IS DFGN)"
    R11 = "IF (Erro IS EZ) AND (derivadaErro IS DEGN) THEN (deltaFrequencia IS DFZ)"
    R12 = "IF (Erro IS EZ) AND (derivadaErro IS DEN) THEN (deltaFrequencia IS DFZ)"
    R13 = "IF (Erro IS EZ) AND (derivadaErro IS DEZ) THEN (deltaFrequencia IS DFZ)"
    R14 = "IF (Erro IS EZ) AND (derivadaErro IS DEP) THEN (deltaFrequencia IS DFZ)"
    R15 = "IF (Erro IS EZ) AND (derivadaErro IS DEGP) THEN (deltaFrequencia IS DFZ)"
    R16 = "IF (Erro IS EP) AND (derivadaErro IS DEGN) THEN (deltaFrequencia IS DFGP)"
    R17 = "IF (Erro IS EP) AND (derivadaErro IS DEN) THEN (deltaFrequencia IS DFP)"
    R18 = "IF (Erro IS EP) AND (derivadaErro IS DEZ) THEN (deltaFrequencia IS DFP)"
    R19 = "IF (Erro IS EP) AND (derivadaErro IS DEP) THEN (deltaFrequencia IS DFP)"
    R20 = "IF (Erro IS EP) AND (derivadaErro IS DEGP) THEN (deltaFrequencia IS DFP)"
    R21 = "IF (Erro IS EGP) AND (derivadaErro IS DEGN) THEN (deltaFrequencia IS DFGP)"
    R22 = "IF (Erro IS EGP) AND (derivadaErro IS DEN) THEN (deltaFrequencia IS DFGP)"
    R23 = "IF (Erro IS EGP) AND (derivadaErro IS DEZ) THEN (deltaFrequencia IS DFGP)"
    R24 = "IF (Erro IS EGP) AND (derivadaErro IS DEP) THEN (deltaFrequencia IS DFGP)"
    R25 = "IF (Erro IS EGP) AND (derivadaErro IS DEGP) THEN (deltaFrequencia IS DFP)"
    
    rules = [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25]
    return rules


def create_fuzzy():
    
    FS = FuzzySystem(show_banner=False)
    
    # Entrada: Erro
    E1 = FuzzySet(function=Trapezoidal_MF(-15, -15, -10, -5), term="EGN")
    E2 = FuzzySet(function=Triangular_MF(-7, -4, -0.5) , term="EN")
    E3 = FuzzySet(function=Triangular_MF(-1.5, 0, 1.5), term="EZ")
    E4 = FuzzySet(function=Triangular_MF(0.5, 4, 7), term="EP")
    E5 = FuzzySet(function=Trapezoidal_MF(5, 10, 15, 15), term="EGP")
    Erro = LinguisticVariable([E1,E2,E3,E4,E5], universe_of_discourse=[-15,15])
    FS.add_linguistic_variable("Erro", LV=Erro)

    # Entrada: Derivada do Erro
    DE1 = FuzzySet(function=Trapezoidal_MF(-5, -5, -3, -1.5), term="DEGN")
    DE2 = FuzzySet(function=Triangular_MF(-2.5, -1, -0.25) , term="DEN")
    DE3 = FuzzySet(function=Triangular_MF(-0.5, 0, 0.5), term="DEZ")
    DE4 = FuzzySet(function=Triangular_MF(0.25, 1, 2.5), term="DEP")
    DE5 = FuzzySet(function=Trapezoidal_MF(1.5, 3, 5, 5), term="DEGP")
    derivadaErro = LinguisticVariable([DE1,DE2,DE3,DE4,DE5], universe_of_discourse=[-5,5])
    FS.add_linguistic_variable("derivadaErro", LV=derivadaErro)
    
    # Saída: Derivada da Frequência
    DF1 = FuzzySet(function=Trapezoidal_MF(-5, -5, -4, -2), term="DFGN")
    DF2 = FuzzySet(function=Triangular_MF(-3, -1.5, -0.25) , term="DFN")
    DF3 = FuzzySet(function=Triangular_MF(-0.5, 0, 0.5), term="DFZ")
    DF4 = FuzzySet(function=Triangular_MF(0.25, 1.5, 3), term="DFP")
    DF5 = FuzzySet(function=Trapezoidal_MF(2, 4, 5, 5), term="DFGP")
    deltaFrequencia = LinguisticVariable([DF1,DF2,DF3,DF4,DF5], universe_of_discourse=[-5,5])
    FS.add_linguistic_variable("deltaFrequencia", LV=deltaFrequencia)
    
    rules = get_rules()
    FS.add_rules(rules)
    
    return FS


def get_results(
    erro: float,
    derivadaErro: float,
    FS: FuzzySystem
) -> float:
    
    FS.set_variable("Erro", erro)
    FS.set_variable("derivadaErro", derivadaErro)
    
    return FS.Mamdani_inference(subdivisions=1000)['deltaFrequencia']