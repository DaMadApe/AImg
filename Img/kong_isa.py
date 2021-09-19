import numpy as np
from skimage import (io, data, color, 
                     util, filters, exposure)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def kong_isa(img, DR=0.7, BR=0.2, blur=10):

    # Aplicar desenfoque con filtro de promedio
    kernel = np.ones((blur, blur))
    BI = filters.rank.mean(img, kernel) # Imagen desenfocada

    # Promedio de iluminación de imagen de entrada
    mean = np.sum(img)/np.size(img) # (3)
    # Separar imagen desenfocada en secciones brillante y oscura
    bright = BI > mean
    dark = BI < mean

    # Bright & Dark Enhancers
    BE = np.zeros(img.shape)
    DE = np.zeros(img.shape)
    # Se asignan así para preservar la forma de la matriz con ceros en huecos
    BE[bright] = (BI[bright] - mean) / mean # (1)
    DE[dark] = (mean - BI[dark]) / mean # (2)

    # Aplicar mejoras a imagen de entrada
    img_en = img + DR*DE*img - BR*BE*img # (4)

    # Incremento de contraste por estiramiento de histograma
    flat_hist = sorted(img.flatten()) # Facilita n-ésimo pixel de histograma
    Lt = 255
    N = img.size # Número de pixeles
    n = int(N/Lt) # (6)
    p = N - n # (7)
    Ln = flat_hist[n]
    Lp = flat_hist[p]

    img_out = (img_en - Ln) * (Lt/(Lp - Ln)) # (5)

    return img_out, BE, DE


"""
Pruebas
"""
img = io.imread('datos/backlit.jpg')
img = util.img_as_ubyte(color.rgb2gray(img)) # Expresar en 0-255
img_eq_he = exposure.equalize_hist(img)*255 # Ecualización convencional
# Método propuesto
DR = 1.2 # Aclaramiento de oscuros
BR = 0.3 # Oscurecimiento de claros
img_eq_kong, BE, DE = kong_isa(img, DR=DR, BR=BR)

# Comparar imágenes
imgs = [(img, 'Imagen original'),
        (img_eq_he, 'Ecualización de histograma'),
        (img_eq_kong, f'Método propuesto, DR={DR}, BR={BR}')]

fig1, ax1 = plt.subplots(len(imgs), 2, figsize=(8, len(imgs)*4))
for i in range(len(imgs)):
    im, title = imgs[i]
    ax1[i, 0].imshow(im, cmap='gray', norm=Normalize(0, 255))
    ax1[i, 0].axis('off')
    ax1[i, 0].set_title(title)
    ax1[i, 1].hist(im.flatten(), bins=256, range=(0, 255))

plt.tight_layout()

# Probar distintos valores de DR y BR
kong_params = [(0.5, 0.5),
               (1.2, 0.5),
               (0.5, 0.2),
               (1.2, 0.2)]

fig2, ax2 = plt.subplots(1, len(kong_params), figsize=(len(kong_params)*4, 4))
for i, params in enumerate(kong_params):
    DR, BR = params
    img_eq, _, _ = kong_isa(img, DR=DR, BR=BR)
    ax2[i].imshow(img_eq, cmap='gray', norm=Normalize(0, 255))
    ax2[i].axis('off')
    ax2[i].set_title(f'DR={DR}, BR={BR}')

plt.tight_layout()
plt.show()