import numpy as np
from Timbiriche import Tablero

class BotAleatorio:
    """
    Agente de juego totalmente aleatorio
    """
    def __init__(self, game):
        self.game = game

    def play(self, board):
        validMoves = self.game.getValidMoves(board, 0)
        validActions = np.argwhere(validMoves).flatten()
        return np.random.choice(validActions)


class BotAleatorioAvaro:
    """
    Agente aleatorio, excepto si hay opción de cerrar
    algún cuadro y anotar
    """
    def __init__(self, game):
        self.game = game

    def play(self, board):
        validMoves = self.game.getValidMoves(board, 0)
        validActions = np.argwhere(validMoves).flatten()

        tablero = Tablero(self.game.n)
        tablero.tablero = np.copy(board)
        for action in validActions:
            if tablero.evalScore(*self.game.actions[action])>0:
                return action

        return np.random.choice(validActions)


class Humano:
    """
    Interfaz para que juegue
    una persona
    """
    def __init__(self, game):
        self.game = game
        print("Timbiriche")

    def play(self, board):
        n = self.game.n
        tablero = Tablero(n)
        tablero.tablero= np.copy(board)
        #print(tablero)
        while True:
            action = None
            print("mov = [arriba, der, izquierda, aba, ...]")
            x, y, mov = input('Tiro (x,y,mov): ').split(',')
            y, x = int(y), int(x)
            if x not in range(n) or y not in range(n):
                print('Coordenadas inválidas')
            else:
                if mov in ['arriba', 'arr']:
                    action = self.game.actions.index((y, x, 1))
                elif mov in ['derecha', 'der']:
                    action = self.game.actions.index((y, x, 2))
                elif mov in ['izquierda', 'izq'] and x>0:
                    action = self.game.actions.index((y, x-1, 2))
                elif mov in ['abajo', 'aba'] and y>0:
                    action = self.game.actions.index((y-1, x, 1))
                else:
                    print('Movimientos válidos: arriba, arr, derecha, izq, etc.')
                    continue
                if self.game.getValidMoves(board, 0)[action] == 0:
                    print('Movimiento inválido')
                    continue
                return action