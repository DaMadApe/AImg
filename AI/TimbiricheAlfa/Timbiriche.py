import numpy as np

from utils import *
from Game import Game
from Tablero import Tablero


class Timbiriche(Game):

    def __init__(self, n):
        self.n = n
        self.score = [0, 0]
        # Conjunto de acciones, para tirar con un solo entero
        self.actions = []         
        for y in range(self.n):
            for x in range(self.n):
                self.actions.append((y,x,1))
                self.actions.append((y,x,2))

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        self.score = [0, 0]
        return Tablero(self.n).tablero

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.n+1, self.n)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 2*self.n**2

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
        nextBoard = Tablero(self.n)
        nextBoard.tablero = np.copy(board)
        p = nextBoard.mover(*self.actions[action])
        if p>1: # Anotación
            self.score[(player+1)//2] += p-1
        if p==1: # Tiro válido sin anotaciones
            nextPlayer = -player
        else: # Tiro fallido o tiro con anotación
            nextPlayer = player
        return nextBoard.tablero, nextPlayer

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
        for action in self.actions:
            is_valid = Tablero.validarExterno(board, *action)
            validMoves.append(int(is_valid))
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
        s1, s2 = self.score
        if s1+s2 == (self.n-1)**2:
            if s1>s2:
                return 1#player
            if s2>s1:
                return -1#-player
            else:
                return 0.001 # Empate
        else:
            return 0 

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
        return board

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
        symts = []

        def rot90(tab, n):
            # Rotación de tablero +90 según mano derecha
            rot_tab = Tablero(n)
            for y in range(n):
                for x in range(n):
                    prev = tab[y,x]
                    if prev == 1:
                        rot_tab[x,n-y-2] += 2
                    if prev == 2:
                        rot_tab[x,n-y-1] += 1
                    if prev == 3:
                        rot_tab[x,n-y-2] += 2
                        rot_tab[x,n-y-1] += 1
            return rot_tab.tablero

        def rot_pol90(pol, n):
            # Rotación correspondiente del vector pi
            polmat = np.reshape(pol, (n, n, 2))
            newmat = np.zeros((n, n, 2))
            for i in range(n):
                for j in range(n):
                    newmat[n-j-1, i, 1] = polmat[i, j, 0]
                    newmat[n-j-2, i, 0] = polmat[i, j, 1]
            return newmat.flatten()

        def reflect(tab, n):
            # Reflejar respecto eje x
            ref_tab = Tablero(n)
            for y in range(n):
                for x in range(n):
                    prev = tab[y,x]
                    if prev == 1:
                        ref_tab[n-y-2, x] += 1
                    if prev == 2:
                        ref_tab[n-y-1, x] += 2
                    if prev == 3:
                        ref_tab[n-y-2, x] += 1
                        ref_tab[n-y-1, x] += 2
            return ref_tab.tablero

        def reflect_pol(pol, n):
            # Reflexión correspondiente de vector pi
            polmat = np.reshape(pol, (n, n, 2))
            newmat = np.zeros((n, n, 2))
            for i in range(n):
                for j in range(n):
                    newmat[n-i-2, j, 0] = polmat[i, j, 0]
                    newmat[n-i-1, j, 1] = polmat[i, j, 1]
            return newmat.flatten()

        tab_rot90 = rot90(board, self.n)
        tab_rot180 = rot90(tab_rot90, self.n)
        tab_rot270 = rot90(tab_rot180, self.n)

        pi_rot90 = rot_pol90(pi, self.n)
        pi_rot180 = rot_pol90(pi_rot90, self.n)
        pi_rot270 = rot_pol90(pi_rot180, self.n)

        tab_ref1 = reflect(board, self.n)
        tab_ref2 = reflect(tab_rot90, self.n)

        pi_ref1 = reflect_pol(pi, self.n)
        pi_ref2 = reflect_pol(pi_rot90, self.n)

        return [(tab_rot90, pi_rot90),
                (tab_rot180, pi_rot180),
                (tab_rot270, pi_rot270),
                (tab_ref1, pi_ref1),
                (tab_ref2, pi_ref2)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return np.array_str(board)