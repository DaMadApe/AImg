import numpy as np

def modelo_random(tablero, score):
    movs_val = 3 - tablero.tablero
    x = np.random.randint(0, tablero.w-1)
    y = np.random.randint(0, tablero.w-1)
    mov = movs_val[y, x]
    if mov==3:
        mov - np.random.randint(0,2)
    return x, y, mov