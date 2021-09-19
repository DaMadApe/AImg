"""
Juego de timbiriche con oponente automático
"""
import numpy as np

class Tablero():

    def __init__(self, w=1, h=10):
        self.tablero=np.zeros((h,w), dtype=np.uint8)
        # Movimientos_válidos = 3-self.tablero

    def __call__(self, x, y):
        # Para que self(x, y) = self.tablero[y, x]
        return self.tablero[y, x]

    def __repr__(self):
        # Para definir lo que devuelve print(self)
        # Se invierte eje vertical al imprimir por conveniencia
        repr = ''
        sym = ['. ', '| ', '._', '|_']
        for fil in self.tablero:
            for pos in fil[::-1]:
                repr = sym[pos] + repr
            repr = '\n' + repr
        return repr

    def mover(self, x, y, mov):
        """
        x: Columna
        y: Fila
        mov: Tiro, 1 para vertical, 2 para horizontal 
        """
        if mov not in [1,2]:
            raise ValueError('Movimientos válidos: 1, 2')
        pos = self.tablero[y, x]
        h, w = self.tablero.shape
        # Validación de tiro
        cond = pos==0  # Caso: no hay líneas previas
        if x < w-1 and y < h-1:
            cond |= (pos+mov==3)  # Caso: la línea previa es compatible
        else:
            # Puntos en la orilla del tablero están restringidos 
            if x == w-1:
                cond &= mov==1 # Sólo se vale ir vertical
            if y == h-1:
                cond &= mov==2 # Sólo se vale ir horizontal
        if cond:
            self.tablero[y, x] += mov
            p = self.score(x, y, mov)
            return 1+p # Turno completo, opcionalmente con puntos
        return 0 # Turno incompleto

    def score(self, x, y, mov):
        """
        Calcular puntaje correspondiente a un tiro (x,y,mov)
        Revisar si la línea cierra algún cuadrado
        """
        p = 0
        if mov==1:
            # pos%2 = (pos==1 or pos==3)
            # pos>1 = (pos==2 or pos==3)
            if self(x-1, y)==3 and self(x-1, y+1)>1:
                p += 1
            if self(x, y)==3 and self(x, y+1)>1 and self(x+1, y)%2:
                p += 1
        else:
            if self(x, y-1)==3 and self(x+1, y-1)%2:
                p += 1
            if self(x, y)==3 and self(x+1, y)%2 and self(x, y+1)>1:
                p += 1
        return p


class Juego():

    def __init__(self, tablero):
        # Puntajes de cada jugador
        self.score_p1 = 0
        self.score_p2 = 0

    

tablero = Tablero()
print(tablero)
tablero.mover(2, 2, 2)
tablero.mover(3, 2, 1)
tablero.mover(2, 2, 1)
#tablero.mover(2, 3, 2)
tablero.mover(3, 3, 1)
tablero.mover(2, 4, 2)
tablero.mover(2, 3, 1)

tablero.mover(0, 0, 1)
tablero.mover(0, 1, 2)
tablero.mover(0, 0, 2)
#tablero.mover(1, 0, 1)
tablero.mover(1, 1, 2)
tablero.mover(2, 0, 1)
tablero.mover(1, 0, 2)

print(tablero)