from PIL import Image
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import random

def salt_pepper_noise(gris,nivel_ruido):
    salida = np.zeros(gris.shape, dtype = np.uint8)
    umbral = 1 - nivel_ruido
    for i in range(gris.shape[0]):
        for j in range(gris.shape[1]):
            ran = random.random()
            if ran < nivel_ruido:
                salida[i][j] = 0
            elif ran > umbral:
                salida[i][j] = 255
            else:
                salida[i][j] = gris[i][j]
    return salida


plt.close('all')

Ima = np.array(Image.open('rostros.jpg').convert('L'))
Ima_ruido = salt_pepper_noise(gris, 0.2)