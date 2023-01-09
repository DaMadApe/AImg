import numpy as np
# Módulos del proyecto
import Arena
from MCTS import MCTS
from Timbiriche import Timbiriche as Game
from Tablero import Tablero
from Jugadores import *
from TimbiricheNet import ModeloNeuronal as NNet
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

def nnetPlayer(g, checkpoint_folder, checkpoint_file):
    net = NNet(g)
    net.load_checkpoint(checkpoint_folder, checkpoint_file)
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, net, args1)
    netp = lambda x: netp.argmax(mcts1.getActionProb(x, temp=0))
    return netp


g = Game(6)
# Ubicación de parámetros almacenados
checkpoint_folder = 'checkpoints/'
checkpoint_file = 'checkpoints_best.pt'

# Catálogo de jugadores
rp = BotAleatorio(g).play
gp = BotAleatorioAvaro(g).play
hp = Humano(g).play
#n1p = nnetPlayer(g, checkpoint_folder, checkpoint_file)
"""
Configuración
"""
player1 = hp
player2 = gp # Jugador avaro

arena = Arena.Arena(player1, player2, g, display=Tablero.display)

print(arena.playGames(4, verbose=True))