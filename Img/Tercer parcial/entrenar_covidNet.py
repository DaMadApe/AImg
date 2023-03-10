import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

from covidNet import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parámetros de entrenamiento
lr = 1e-3
batch_size = 32
epochs = 3

"""
Datasets
"""
train_transforms = transforms.Compose([
    transforms.RandomCrop(260),
    transforms.ToTensor(),
    transforms.Grayscale(),
    #transforms.Normalize(0.5094, 0.2503)
    transforms.Normalize(0.5366, 0.2207) # Con Randomcrop (260)
    ])

eval_transforms = transforms.Compose([
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Grayscale(),
    #transforms.Normalize(0.5094, 0.2503)
    transforms.Normalize(0.5366, 0.2207) # Con Randomcrop (260)
    ])

# ImageFolder automatiza carga de imágenes y etiqueta según folder
train_set = ImageFolder('datos/pulmones/train', train_transforms)
val_set = ImageFolder('datos/pulmones/val', eval_transforms)

# Generadores para iterar los datos
train_loader = DataLoader(train_set,
                            batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set,
                        batch_size=batch_size, shuffle=True)

"""
Inicializar modelo
"""
model = CovidNet2()
# Revisar si hay checkpoints de entrenamientos anteriores
checkpoint_path = 'modelos/covidNet.pt'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))

"""
Entrenamiento
"""
criterio = torch.nn.CrossEntropyLoss()
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