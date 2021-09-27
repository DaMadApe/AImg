from Game import Game
from NeuralNet import NeuralNet
import numpy as np

from TimbiricheNet import TimbiricheNet


class Tablero():
    """
    Representación del tablero de juego, incluyendo métodos
    para tirar en el tablero y 
    """

    def __init__(self, w=5, h=5):
        self.tablero = np.zeros((h, w), dtype=np.uint8)
        self.w = w
        self.h = h
        self.score = [0, 0]
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


class Timbiriche(Game):

    def __init__(self, tablero):
        self.tablero = tablero
        self.h = tablero.h
        self.w = tablero.w

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        tablero = np.zeros((self.h, self.w), dtype=np.uint8)
        return tablero

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.h, self.w)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        # valid_movs = 3 - self.tablero.tablero
        # size = sum([2 if m==3 else 0 for m in valid_movs.flatten()])
        # size += sum([1 if m in [1,2] else 0 for m in valid_movs.flatten()])
        # size += - self.h - self.w

        size = 2 * (self.h-1) * (self.w-1) + self.h + self.w
        return size

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        nextBoard = Tablero(self.w, self.h)
        nextBoard.tablero = board.tablero.copy()
        p = nextBoard.move(action)
        if p==1: # Tiro válido sin anotaciones
            nextPlayer = -player
        else:
            nextPlayer = player
        return nextBoard, nextPlayer

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        validMoves = []
        for y, row in enumerate(board.tablero):
            for x, pos in enumerate(row):
                validMoves.append(board.validar(y,x,1))
                validMoves.append(board.validar(y,x,2))

        return validMoves

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
        """
        tablero_lleno = 3*(self.h-1)*(self.w-1) + 2*(self.w-1) + (self.h-1)
        np.sum(self.tablero.tablero) == tablero_lleno
        
        pass

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board.tablero

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """

        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tablero.toString()


class ModeloNeuronal(NeuralNet):
    """
    Este es un envoltorio de la clase TimbiricheNet, que
    sirve de interfaz entre el modelo y el resto del
    programa. Esta clase sigue los prototipos de
    la clase NeuralNet, y de ahí vienen
    las siguientes descripciones
    """
    def __init__(self, game):
        self.net = TimbiricheNet

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        self.net.train()

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        self.net.predict(board)

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        self.net.save(folder, filename)

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        self.net.load(folder, filename)