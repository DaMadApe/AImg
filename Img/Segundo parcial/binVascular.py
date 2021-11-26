import numpy as np
import matplotlib.pyplot as plt
from skimage import (io, data, exposure, util,
                     filters, morphology)
from skimage.morphology import disk
from skimage.filters import rank 


# Cargar imagen
img = io.imread('datos/antebrazo/Venas4.jpg')
#img = io.imread('datos/ojo/03_L.jpg')
#img = data.retina()


# Pasos del procesamiento
proceso = []

img_proc = img[:,:,1]
proceso.append((img_proc.copy(), 'Canal Verde Invertido'))

img_proc = util.img_as_ubyte(exposure.equalize_adapthist(img_proc))
proceso.append((img_proc.copy(), 'Ecualización adaptativa'))

img_proc = util.img_as_ubyte(img_proc < rank.otsu(img_proc, disk(12)))
proceso.append((img_proc.copy(), 'Umbral local de Otsu r=12'))

img_proc = rank.mean(img_proc, disk(2))
proceso.append((img_proc.copy(), 'Filtro medio r=2'))

img_proc = util.img_as_ubyte(img_proc>200)
proceso.append((img_proc.copy(), 'Umbral >180'))

img_proc = morphology.area_opening(img_proc, 200)
proceso.append((img_proc.copy(), 'Area opening A>200'))


# # Mostrar imágenes
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