import numpy as np


class Tablero():
    """
    Representación del tablero de juego, incluyendo métodos
    para tirar en el tablero y validar los movimientos posibles.
    """

    def __init__(self, n=5):
        self.tablero = np.zeros((n, n), dtype=np.uint8)
        self.n = n
        self.score = [0, 0]
        # Movimientos_válidos = 3-self.tablero

    def __getitem__(self, idx):
        return self.tablero[idx]

    def __repr__(self):
        # Para definir lo que devuelve print(self)
        # Se invierte eje vertical al imprimir por conveniencia
        repr = '\n '
        for i in range(self.tablero.shape[1]):
            repr += str(i) + ' '
        sym = ['. ', '| ', '._', '|_']
        for j, fil in enumerate(self.tablero):
            for pos in fil[::-1]:
                repr = sym[pos] + repr
            repr = f'\n{j}' + repr
        return repr

    def validarTiro(self, y, x, mov):
        pos = self[y, x]
        # Validación de tiro
        cond = mov in [1, 2]
        if x < self.w-1 and y < self.h-1:
            cond &= (pos == 0 or (pos+mov == 3))
        else:
            # Puntos en la orilla del tablero están restringidos
            if x == self.w-1:
                cond &= mov == 1  # Sólo se vale ir vertical
            if y == self.h-1:
                cond &= mov == 2  # Sólo se vale ir horizontal
        return cond

    def mover(self, y, x, mov):
        """
        x: Columna
        y: Fila
        mov: Tiro, 1 para vertical, 2 para horizontal 
        """
        if self.validarTiro(y, x, mov):
            self.tablero[y, x] += mov
            p = self.evalScore(y, x, mov)
            return 1 + p  # Turno completo + los puntos ganados
        return 0  # Turno incompleto

    def evalScore(self, y, x, mov):
        """
        Calcular puntaje correspondiente a un tiro (y,x,mov)
        Revisar si la línea cierra algún cuadrado
        """
        p = 0
        if mov == 1:
            # pos%2 = (pos==1 or pos==3)
            # pos>1 = (pos==2 or pos==3)
            if self[y, x-1] == 3 and self[y+1, x-1] > 1:
                p += 1
            if self[y, x] == 3 and self[y, x+1] > 1 and self[y, x+1] % 2:
                p += 1
        else:
            if self[y, x-1] == 3 and self[y-1, x+1] % 2:
                p += 1
            if self[y, x] == 3 and self[y, x+1] % 2 and self[y, x+1] > 1:
                p += 1
        return p