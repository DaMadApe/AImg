import numpy as np
import matplotlib.pyplot as plt

vDec = [0.1458, 0.0938 ,0.1493,
        0.0938, 0.1667, 0.0914,
        0.1493, 0.0914, 0.1528]

cParo = 1

while cParo == 1:
    tablero = np.zeros(9)
    turno = int(input("¿Quién inicia? 1: Máquina, 2: Persona"))
    