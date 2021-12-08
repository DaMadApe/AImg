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
            nn.Softmax())

    def forward(self, x):
        return self.model(x)


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
            nn.Softmax())

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    import os
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from tqdm import tqdm

    path = 'datos/pulmones/'
    c1 = 'COVID'
    c2 = 'Neumonía'
    c3 = 'Sano'

    # Parámetros de entrenamiento
    lr = 3e-4
    batch_size = 32
    epochs = 3

    """
    Datasets
    """

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale()
        ])

    # ImageFolder automatiza carga de imágenes y etiqueta según folder
    train_set = ImageFolder('datos/pulmones/train', img_transforms)
    val_set = ImageFolder('datos/pulmones/val', img_transforms)

    # Generadores para iterar los datos
    train_loader = DataLoader(train_set,
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size, shuffle=True)

    """
    Entrenamiento
    """

    # Instancia del modelo
    model = CovidNet()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parámetros totales: {total_params}')
    
    # Revisar si hay checkpoints de entrenamientos anteriores
    checkpoint_path = 'modelos/covidNet.pt'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(path))


    criterio = nn.CrossEntropyLoss()
    optimizador = torch.optim.Adam(model.parameters(), lr=lr)

    # tqdm sólo es para mostrar barra de progreso
    progress = tqdm(range(epochs), desc='Training')

    for _ in progress:
        # Paso de entrenamiento
        model.train()
        for X, Y in train_loader:
            pred = model(X)
            loss = criterio(pred, Y)
            optimizador.zero_grad()
            loss.backward()
            optimizador.step()

        # Paso de validación
        model.eval() 
        with torch.no_grad():
            for X, Y in val_loader:
                pred = model(X)
                val_loss = criterio(pred, Y)

        # Imprimir últimos valores de época en la barra de progreso
        progress.set_postfix(Loss=loss.item(), Val=val_loss.item())

    # Guardar parámetros actuales en el checkpoint
    # La carpeta ya debe existir o da error
    torch.save(model.state_dict(), checkpoint_path)