import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
# Módulos del proyecto
from NeuralNet import NeuralNet
from utils import *


# Argumentos de entrenamiento
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
})

class TimbiricheNet(nn.Module):
    """
    Definición de la arquitectura de la red neuronal
    y su método de entrenamiento.
    """
    def __init__(self, game, args):
        super(TimbiricheNet, self).__init__()

        self.n, _ = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.convnet = nn.Sequential(
            nn.Conv2d(1, args.num_channels, 3),
            nn.BatchNorm2d(args.num_channels),#nn.MaxPool2d(),
            nn.ReLU(),
            nn.Conv2d(args.num_channels, args.num_channels, 3),
            nn.BatchNorm2d(args.num_channels),
            nn.ReLU())

        self.p_head = nn.Sequential(
            nn.Conv1d(args.num_channels, args.num_channels, 3),
            nn.BatchNorm1d(args.num_channels),
            nn.Linear(args.num_channels, 64),
            nn.ReLU(),
            nn.Linear(64 ,self.action_size),
            nn.Softmax()) # Activación final restringe salida [0,1] con suma = 1

        self.v_head = nn.Sequential(
            nn.Conv1d(args.num_channels, args.num_channels, 3),
            nn.BatchNorm1d(args.num_channels),
            nn.Linear(args.num_channels, 64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Tanh())

    def forward(self, s):
        s = self.convnet(s)
        pi = self.p_head(s)
        v = self.v_head(s)
        return pi, v


class ModeloNeuronal(NeuralNet):
    """
    Este es un envoltorio de la clase TimbiricheNet, que
    sirve de interfaz entre el modelo y el resto del
    programa. Esta clase sigue los prototipos de la clase
    NeuralNet, y de ahí vienen las siguientes descripciones
    """
    def __init__(self, game):
        self.net = TimbiricheNet(game, args)

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
        optimizer = optim.Adam(self.net.parameters())
        for epoch in range(args.epochs):
            print(f'--- Época: {epoch+1} ---')

            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # Acomodar elementos en memoria
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # Calcular las pérdidas o errores
                out_pi, out_v = self.nnet(boards) # Forward

                criterio_pi = nn.NLLLoss()
                loss_pi = criterio_pi(out_pi, target_pis)
                criterio_v = nn.MSELoss()
                loss_v = criterio_v(out_v, target_vs)

                loss = loss_pi + loss_v

                # Registrar pérdidas
                pi_losses.update(loss_pi.item(), boards.size(0))
                v_losses.update(loss_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
        
                # Calcular gradiente y propagarlo
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        board = torch.FloatTensor(board.astype(torch.float64))
        if args.cuda:
            # Acomodar el tensor eficientemente en memoria de GPU
            board = board.contiguous().cuda()
        board = board.view(1, self.n, self.n)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

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
            'state_dict': self.net.state_dict(),
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