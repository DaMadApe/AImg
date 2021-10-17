import numpy as np
import matplotlib.pyplot as plt
from skimage import io, data, color, util, exposure
from skimage.filters import threshold_multiotsu
import skfuzzy as fuzzy
from crecimientoReg import (registrar_semillas,
                            aislar_region)


def fcm_img(img, c):
    # Fuzzy c-means para imagen de 1 canal
    datos = np.reshape(img, (1, -1))
    fcm_result = fuzzy.cmeans(datos,
                              c=c,
                              m=2,
                              error=5e-4,
                              maxiter=15)
    cntr, u = fcm_result[0:2]
    clusters = []
    for cluster in u:
        clusters.append(np.reshape(cluster, img.shape))
    return cntr, clusters

def img_reg(img, region):
    out = np.ones(img.shape)
    out[region] = img[region]
    return out

def multi_imshow(imgs):
    # Auxiliar para imprimir lista de imágenes
    plt.figure(figsize=(4*len(imgs), 4))
    for i, (img, title, cmap) in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()


"""
Pruebas
"""
img = data.retina()
img = io.imread('datos/Brain.png')


c = 3

# Método Otsu múltiple
thres = threshold_multiotsu(img, classes=c)
img_otsu = np.digitize(img, bins=thres)
regs_otsu = [img_otsu==i for i in range(c)]

imgs = [(img_otsu, 'Umbral de Otsu', 'jet')]
for i in range(c):
    imgs.append((regs_otsu[i], f'Región {i}', 'gray'))

multi_imshow(imgs)


# Método FCM
img_fcm, clust_fcm = fcm_img(img, c)

imgs = []
for i in range(c):
    imgs.append((clust_fcm[i], f'Cluster {i}', 'gray'))

multi_imshow(imgs)

plt.show()


# Método crecimiento de región
seed = registrar_semillas(img)
reg = aislar_region(img, seed, tolerancia=0.1, visual=True)