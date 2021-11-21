import numpy as np
import matplotlib.pyplot as plt
from skimage import (io, data, exposure, util, color,
                     filters, morphology, transform)

# Cargar imagen
img = io.imread('datos/craneo_3d/IMG_0745 (103).jpg')

# Pasos de segmentación
proceso = []
# Extracción de canal azul
img_proc = color.rgb2gray(img)
# Recorte manual de área de cráneo
img_proc = img_proc[250:850]
proceso.append((img_proc.copy(), 'Canal Verde Invertido'))

img_proc = util.img_as_ubyte(exposure.equalize_adapthist(img_proc))
proceso.append((img_proc.copy(), 'Ecualización adaptativa'))

img_proc = util.img_as_ubyte(img_proc < 50)
proceso.append((img_proc.copy(), 'Umbral local de Otsu r=12'))


# Mostrar imágenes
plt.imshow(img)
plt.title('Imagen original')

plt.figure(figsize=(15, 10))
for i, (img, titulo) in enumerate(proceso):
    plt.subplot(2,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'{i+1}. {titulo}')

plt.tight_layout()
plt.show()