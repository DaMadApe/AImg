import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import tqdm

class Clasificador(nn.Module):

    def __init__(self, n_clases):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(0, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_clases),
            nn.Softmax())

    def forward(self, x):
        return self.model(x)


class ImgsPulmones(Dataset):

    def __init__(self, path, c1, c2, c3, modo):
        self.nc1 = self._contar_archivos(os.path.join(path, c1))
        self.nc2 = self._contar_archivos(os.path.join(path, c2))
        self.nc3 = self._contar_archivos(os.path.join(path, c3))

        self.data = #[path, label]
    
    def _contar_archivos(dir):
        return len([1 for x in list(os.scandir(dir)) if x.is_file()])

    def __len__(self):
        return self.nc1 + self.nc2 + self.nc3
    
    def __getitem__(self, idx):
        if idx < self.nc1
        return img, target


if __name__ == '__main__':

    path = 'datos/pulmones/'
    c1 = 'COVID'
    c2 = 'Neumonía'
    c3 = 'Sano'

    train_set = ImgsPulmones(path, c1, c2, c3, 'train')

    """
    Entrenamiento
    """
    # Parámetros
    lr = 3e-4
    batch_size = 32
    epochs = 5 

    # Generadores para iterar los datos
    train_loader = DataLoader(train_set,
                              batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set,
    #                         batch_size=batch_size, shuffle=True)

    # Instancia del modelo
    model = Clasificador(3)
    
    # Revisar si hay checkpoints de entrenamientos anteriores
    checkpoint_path = 'modelos/covidNet.pt'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(path))


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # tqdm sólo es para mostrar barra de progreso
    progress = tqdm(range(epochs), desc='Training')

    for _ in progress:
        # Train step
        for X, Y in train_loader:
            model.train() # Sacar de for
            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Val step
        # with torch.no_grad():
        #     for X, Y in val_loader:
        #         model.eval()  # Sacar de for
        #         pred = model(X)
        #         val_loss = criterion(pred, Y)

        progress.set_postfix(Loss=loss.item(), Val=val_loss.item())

    torch.save(model.state_dict(), checkpoint_path)