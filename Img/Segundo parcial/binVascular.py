import numpy as np
import matplotlib.pyplot as plt
from skimage import (io, data, exposure, util,
                     filters, morphology)
from skimage.morphology import disk
from skimage.filters import rank 


def pasabajos_ideal(shape, f_corte):
    fil, col = shape
    y, x = np.ogrid[-fil/2:fil/2, -col/2:col/2]
    filtro = (x**2 + y**2 <= f_corte**2)
    return filtro.astype(float)

def pasaaltos_ideal(shape, f_corte):
    fil, col = shape
    y, x = np.ogrid[-fil/2:fil/2, -col/2:col/2]
    filtro = (x**2 + y**2 >= f_corte**2)
    return filtro.astype(float)

def aplicar_filtro_frec(img, kernel):
    if kernel.shape == img.shape:
        filtro = kernel
    else:
        filtro = np.zeros(img.shape)
        filtro[:kernel.shape[0], :kernel.shape[1]] = kernel

    img_frec = np.fft.fftshift(np.fft.fft2(img))
    filtro = pasaaltos_ideal(img.shape, 100)

    # Aplicar el filtro en frecuencia
    convolucion = img_frec * filtro
    # Regresar al dominio espacial
    img_ift = np.fft.ifft2(np.fft.fftshift(convolucion))
    img_final = np.abs(img_ift)
    img_final = util.img_as_ubyte(img_final/np.max(img_final))
    return img_final


# Cargar imagen
img = io.imread('datos/ojo/03_L.jpg')
#img = data.retina()


# Pasos del procesamiento
proceso = []

img_proc = img[:,:,1]
proceso.append((img_proc.copy(), 'Canal Verde Aislado'))

img_proc = util.img_as_ubyte(exposure.equalize_adapthist(img_proc))
proceso.append((img_proc.copy(), 'Ecualización apaptativa'))

img_proc = util.img_as_ubyte(img_proc < rank.otsu(img_proc, disk(30)))
proceso.append((img_proc.copy(), 'Umbral localizado de Otsu'))

img_proc = rank.median(img_proc, disk(15))
proceso.append((img_proc.copy(), 'med mediano'))

#img_final = img_med > filters.threshold_otsu(img_med)
img_proc = morphology.opening(img_proc/255, disk(15))
proceso.append((img_proc.copy(), 'Opening a imagen umbral'))


# Mostrar imágenes
plt.figure(figsize=(15, 8))
plt.subplot(2,3,1)
plt.imshow(img)
plt.axis('off')
plt.title('Imagen original')

for i, (img, titulo) in enumerate(proceso):
    plt.subplot(2,3,i+2)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'{i+1}. {titulo}')

plt.tight_layout()
plt.show()