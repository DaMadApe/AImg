import Arena
from MCTS import MCTS
from Timbiriche import Timbiriche as Game
from Timbiriche import Tablero
from Jugadores import *
from TimbiricheNet import ModeloNeuronal as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = Game(3)


# all players
rp = BotAleatorio(g).play
gp = BotAleatorioAvaro(g).play
hp = Humano(g).play



# nnet players
#n1 = NNet(g)
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
#mcts1 = MCTS(g, n1, args1)
#n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
n1p = gp #Humano(g).play#

if human_vs_cpu:
    player2 = Humano(g).play
# else:
#     n2 = NNet(g)
#     n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
#     args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
#     mcts2 = MCTS(g, n2, args2)
#     n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

#     player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.


arena = Arena.Arena(n1p, player2, g, display=Tablero.display)

print(arena.playGames(4, verbose=True))