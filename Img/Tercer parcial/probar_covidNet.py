import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt

from covidNet import *

input_trans = transforms.Compose([
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Grayscale(),
    #transforms.Normalize(0.5094, 0.2503)
    transforms.Normalize(0.5366, 0.2207) # Con Randomcrop (260)
    ])

model = CovidNet2()
checkpoint_path = 'modelos/covidNet.pt'
model.load_state_dict(torch.load(checkpoint_path))

val_set = ImageFolder('datos/pulmones/test', input_trans)
val_loader = DataLoader(val_set,
                        batch_size=1, shuffle=True)
img, label = next(iter(val_loader))
model.eval()
pred = model(img)
real = ['Covid', 'Pneumonía', 'Sano'][label]

print(f"""Predicción:
    Covid: {pred[0,0]}
    Pneum: {pred[0,1]}
    Sano: {pred[0,2]}
    \n Real: {real}""")

img_view = torch.permute(img[0], (1,2,0))
plt.imshow(img_view, cmap='gray')
plt.show()