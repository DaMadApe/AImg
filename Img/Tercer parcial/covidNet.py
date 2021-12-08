import torch
from torch import nn

class CovidNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.ReLU(),
            #nn.BatchNorm2d(32),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),
            # Convolución 1x1 para reducir número de canales
            # Baja número de parámetros de ~22M a ~6M
            # nn.Conv2d(64, 16, (1, 1), padding='same'),
            # nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
            # nn.Linear(21904, 256),
            nn.Linear(87616, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3), # 3 clases
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.model(x)


class CovidNet2(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 8, (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(10952, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3), # 3 clases
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    model = CovidNet()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parámetros totales: {total_params}')

    model = CovidNet2()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parámetros totales: {total_params}')