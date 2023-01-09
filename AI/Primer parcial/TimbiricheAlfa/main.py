import logging

#import coloredlogs

from Coach import Coach
from Timbiriche import Timbiriche as Game
from TimbiricheNet import ModeloNeuronal as nn
from utils import *

log = logging.getLogger(__name__)

#coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.
logging.basicConfig(level=logging.DEBUG)

args = dotdict({
    'numIters': 100,
    'numEps': 64,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 5000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 16,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 24,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': 'AI/TimbiricheAlfa/checkpoints/',
    'load_model': False,
    'load_folder_file': ('checkpoints/','best.pt'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(4)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()