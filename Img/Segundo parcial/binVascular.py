import numpy as np
import matplotlib.pyplot as plt
from skimage import (io, data, exposure, util,
                     filters, morphology)


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
    filtro = pasaaltos_ideal(img_g.shape, 100)

    # Aplicar el filtro en frecuencia
    convolucion = img_frec * filtro
    # Regresar al dominio espacial
    img_ift = np.fft.ifft2(np.fft.fftshift(convolucion))
    # Acondicionar como imagen visible
    # venas = np.array(np.abs(img_ift), dtype=np.ubyte)
    return np.abs(img_ift)


img = io.imread('datos/ojo/03_L.jpg')
#img = data.retina()
img_g = img[:,:,1]

img_eq = util.img_as_ubyte(exposure.equalize_adapthist(img_g))
#img_eq = util.img_as_ubyte(filters.rank.equalize(img_g, morphology.disk(5)))

img_bin = util.img_as_ubyte(img_eq > filters.rank.otsu(img_eq, morphology.disk(30)))


img_med = filters.rank.median(img_bin, morphology.disk(15))

img_final = img_med > filters.threshold_otsu(img_med)


"""
Procesar en frecuencia
"""
filtro = pasabajos_ideal(img_bin.shape, 100)
venas = aplicar_filtro_frec(img_bin, filtro)

# Mostrar imágenes
plt.subplot(2,3,1)
plt.imshow(img)
plt.title('1. Imagen original')

plt.subplot(2,3,2)
plt.imshow(img_g, cmap='gray')
plt.title('2. Canal verde aislado')

plt.subplot(2,3,3)
plt.imshow(img_eq, cmap='gray')
plt.title('3. Ecualización adaptativa')

plt.subplot(2,3,4)
plt.imshow(img_bin, cmap='gray')
plt.title('4. Umbral Otsu local')

plt.subplot(2,3,5)
plt.imshow(img_med, cmap='gray')
plt.title('5. Filtro mediano')

plt.subplot(2,3,6)
plt.imshow(img_final, cmap='gray')
plt.title('6. Umbral Otsu')

plt.axis('off')
plt.tight_layout()
plt.show()

# from crecimientoReg import (registrar_semillas,
#                             aislar_region)
# # Método crecimiento de región
# seed = registrar_semillas(img_bin2)
# reg = aislar_region(img_bin2, seed, tolerancia=0.15, visual=False)