"""
Juego de timbiriche con oponente automático
"""
import numpy as np
from modelo_cpu import modelo_random

class Tablero():
    """
    Representación del tablero de juego, incluyendo métodos
    para tirar en el tablero y 
    """
    def __init__(self, w=5, h=5):
        self.tablero=np.zeros((h,w), dtype=np.uint8)
        self.w = w
        self.h = h
        # Movimientos_válidos = 3-self.tablero

    def __call__(self, y, x):
        # Para que self(y, x) = self.tablero[y, x]
        return self.tablero[y, x]

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
        pos = self.tablero[y, x]
        h, w = self.tablero.shape
        # Validación de tiro
        cond = mov in [1, 2]
        if x < w-1 and y < h-1:
            cond &= pos == 0 or (pos+mov == 3)
        else:
            # Puntos en la orilla del tablero están restringidos
            if x == w-1:
                cond &= mov == 1  # Sólo se vale ir vertical
            if y == h-1:
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
            p = self.score(y, x, mov)
            return 1 + p  # Turno completo + los puntos ganados
        return 0  # Turno incompleto

    def score(self, y, x, mov):
        """
        Calcular puntaje correspondiente a un tiro (y,x,mov)
        Revisar si la línea cierra algún cuadrado
        """
        p = 0
        if mov == 1:
            # pos%2 = (pos==1 or pos==3)
            # pos>1 = (pos==2 or pos==3)
            if self(y, x-1) == 3 and self(y+1, x-1) > 1:
                p += 1
            if self(y, x) == 3 and self(y, x+1) > 1 and self(y, x+1) % 2:
                p += 1
        else:
            if self(y, x-1) == 3 and self(y-1, x+1) % 2:
                p += 1
            if self(y, x) == 3 and self(y, x+1) % 2 and self(y, x+1) > 1:
                p += 1
        return p


class Juego():

    def __init__(self, tablero, modo, modelo_cpu=None, repr=True):
        # Puntajes de cada jugador
        self.tablero = tablero
        self.modelo = modelo_cpu
        self.score = [0, 0]
        if modo == 'CC': # Cpu vs Cpu
            self.juego(self.tiro_cpu, self.tiro_cpu, repr)
        elif modo == 'CP': # Cpu vs Persona
            self.juego(self.tiro_cpu, self.tiro_persona, repr)
        elif modo == 'PC': #Sólo cambia orden de tiro respecto a CP
            self.juego(self.tiro_persona, self.tiro_cpu, repr)
        elif modo == 'PP': # Persona vs persona
            self.juego(self.tiro_persona, self.tiro_persona, repr)
        else:
            raise ValueError('Modos válidos: PP, CP, PC, CC')

    def tiro_cpu(self):
        y, x, mov = self.modelo(self.tablero, self.score)
        tiro = self.tablero.mover(y, x, mov)
        return tiro

    def tiro_persona(self):
        y, x, mov = map(int, input('Tiro (y,x,mov): ').split(','))
        if x not in range(self.tablero.w) or y not in range(self.tablero.h):
            print('Coordenadas inválidas')
        elif mov not in [1,2]:
            print('Movimientos válidos: 1, 2')
        else:
            tiro = self.tablero.mover(y, x, mov)
            return tiro
        return 0

    def juego(self, tiro1, tiro2, repr):
        # Calcular equivalente de un tablero lleno de líneas
        h, w = self.tablero.tablero.shape
        tablero_lleno = 3*(h-1)*(w-1) + 2*(w-1) + (h-1)
        if repr:
            print(self.tablero)
            print("mov=1 para arriba, mov=2 para derecha")
            print("Jugador 1")
        while(np.sum(self.tablero.tablero) < tablero_lleno):
            p1, p2 = (0, 0)
            while(p1 != 1): # Sigue tirando si fue un tiro inválido o si anotó
                p1 = tiro1()
                if(p1 > 1):
                    self.score[0] += p1-1
                    if repr:
                        print(self.tablero)
                if np.sum(self.tablero.tablero) == tablero_lleno:
                    break
            if repr:
                print(self.tablero)
                print("Jugador 2")

            while(p2 != 1):
                p2 = tiro2()
                if(p2 > 1):
                    self.score[1] += p2-1
                    if repr:
                        print(self.tablero)
                if np.sum(self.tablero.tablero) == tablero_lleno:
                    break
            if repr:
                print(self.tablero)
                print("Jugador 1")

        if repr:
            print('Fin de juego')
            print(f'Puntaje final: {self.score[0]}-{self.score[1]}')



tablero = Tablero(w=3, h=2)
juego = Juego(tablero, modo='PP', modelo_cpu=modelo_random)