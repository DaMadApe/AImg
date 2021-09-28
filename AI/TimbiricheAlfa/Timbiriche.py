import os
import numpy as np
import torch

from utils import *
from Game import Game
from TimbiricheLogic import Tablero
from NeuralNet import NeuralNet
from TimbiricheNet import TimbiricheNet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class Timbiriche(Game):

    def __init__(self, n):
        self.n

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        tablero = np.zeros((self.n, self.n), dtype=np.uint8)
        return tablero

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.n, self.n)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        size = 2*(self.n-1)**2 + 2*self.n
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
        nextBoard = Tablero(self.n)
        nextBoard.tablero = np.copy(board)
        p = nextBoard.move(action)
        if p==1: # Tiro válido sin anotaciones
            nextPlayer = -player
        else:
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
        tablero = Tablero(self.n)
        tablero.tablero = np.copy(board)
        validMoves = []
        for y in range(self.n):
            for x in range(self.n):
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
        tablero_lleno = 3*(self.n-1)**2 + 3*(self.n-1)
        r = np.sum(board) == tablero_lleno
        return r

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

        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.toString()


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

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        Loads parameters of the neural network from folder/filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.net.load_state_dict(checkpoint['state_dict'])