from skimage import data, color
import numpy as np
import matplotlib.pyplot as plt

"""
Problema 1.1
Procesar en frecuencia para mejorar contraste

Se utiliza un filtro pasabandas para aislar las venas,
y se usa aditivamente este resultado para incrementar
el contraste en la imagen original
"""

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


img = data.retina()
# Trabajar con el canal verde
img_g = img[:,:,1]

img_frec = np.fft.fftshift(np.fft.fft2(img_g))

# Construir un filtro pasabanda
filtro = np.logical_and(pasabajos_ideal(img_g.shape, 500),
                        pasaaltos_ideal(img_g.shape, 10))
# Aplicar el filtro en frecuencia
convolucion = img_frec * filtro
# Regresar al dominio espacial
img_ift = np.fft.ifft2(np.fft.fftshift(convolucion))
# Acondicionar como imagen visible
venas = np.array(np.abs(img_ift), dtype=np.ubyte)

# Construir imagen de salida a partir de original y procesado
img_out = img_g - venas

# Visualizar resultado
plt.figure(figsize=(8, 4))

plt.subplot(1,2,1)
plt.imshow(img_g, cmap='gray')

plt.subplot(1,2,2)
plt.imshow(img_out, cmap='gray')

plt.show()


"""
Problema 1.2
Segmentación del nervio óptico

Se usa crecimiento de región para aislar el nervio.
La selección de semillas se adjunta con los resultados
"""
from crecimientoReg import (registrar_semillas,
                            aislar_region)
# Método crecimiento de región
seed = registrar_semillas(img_out)
reg = aislar_region(img_out, seed, tolerancia=0.15, visual=True)

# Marcar de rojo la región detectada
img_mark = color.gray2rgb(img_g)
img_mark[reg,0] = 255

plt.figure()
plt.imshow(img_mark)
plt.show()