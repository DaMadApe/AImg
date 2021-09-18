import numpy as np
from skimage import (io, data, color, 
                     util, filters, exposure)
import matplotlib.pyplot as plt

def hist_stretch(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return (img-img_min) * 255/(img_max-img_min)

def kong_isa(img, DR=0.7, BR=0.2, blur=5):

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
    BE[bright] = (BI[bright] - mean) / mean # (1)
    DE[dark] = (mean - BI[dark]) / mean # (2)

    # Aplicar mejoras a imagen de entrada
    img_en = img + DR*DE*img - BR*BE*img # (4)

    # Incremento de contraste por estiramiento
    flat_hist = sorted(img.flatten()) # Facilita n-ésimo pixel de histograma
    Lt = 255
    N = img.size # Número de pixeles
    n = int(N/Lt) # (6)
    p = N - n # (7)
    Ln = flat_hist[n]
    Lp = flat_hist[p]

    #img_out = (img_en - Ln) * (Lt/(Lp - Ln)) # (5)
    img_out = hist_stretch(img_en)

    return img_out



"""
Pruebas
"""
img = data.camera()
img = io.imread('datos/backlit.jpg')
img = util.img_as_ubyte(color.rgb2gray(img))
img_eq_he = exposure.equalize_hist(img)*255
img_eq_kong = kong_isa(img, DR=1.2, BR=0.1)

# Comparar imágenes
imgs = [(img, 'Imagen original'),
        (img_eq_he, 'Ecualización de histograma'),
        (img_eq_kong, 'Método propuesto')]

fig1, ax1 = plt.subplots(len(imgs), 2)
for i in range(len(imgs)):
    im, title = imgs[i]
    ax1[i, 0].imshow(im, cmap='gray')
    ax1[i, 0].axis('off')
    ax1[i, 0].set_title(title)
    ax1[i, 1].hist(im.flatten(), bins=256, range=(0, 255))

plt.tight_layout()

# Probar distintos valores de DR y BR
kong_params = [(0.7, 0.2),
               (0.8, 0.2),
               (0.7, 0.3),
               (0.8, 0.3)]

plt.tight_layout()
plt.show()