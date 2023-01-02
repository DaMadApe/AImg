import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform, morphology, filters

"""
Funciones asociadas a proyecciones de rayos en abanico
"""

def trans_fan(imagen, paso_alfa, paso_beta, D, ciclo=360):
    """
    Obtener la proyección en abanico de una imagen

    paso_alfa: Desplazamiento angular de fuente (grados)
    paso_beta: Separación angular de rayos (grados)
    D: Distancia de la fuente
    ciclo: Rotación total de la fuente
    """
    img_pad = np.pad(imagen, D) # Distancia de fuente se mete como padding
    h, w = img_pad.shape
    hm = h//2
    beta_lim = np.arctan(h/(2*w)) # Ángulo de las esquinas contrarias
    pasos_beta = int(2*beta_lim / (paso_beta * np.pi/180))
    pasos_alfa = int(ciclo/paso_alfa)

    proyeccion = np.zeros((pasos_alfa, pasos_beta))

    angulos_alfa = np.arange(0, ciclo, paso_alfa) # alfa se usa en grados
    angulos_beta = np.linspace(-beta_lim, beta_lim, pasos_beta)

    # Barrido de la fuente
    for a, alfa in enumerate(angulos_alfa):
        img_rot = transform.rotate(img_pad, alfa)
        # Abanico
        for b, beta in enumerate(angulos_beta):
            x = np.arange(0, w)
            y = np.linspace(0, w*np.tan(beta)-1, w, dtype=int) + hm
            proyeccion[a, b] = img_rot[y, x].sum()
    return proyeccion

def fan2paralelo(img_fan, D):
    """
    Convertir proyecciones en abanico a proyecciones en paralelo
    """
    a, b = img_fan.shape
    paralelo = np.zeros((a+b, D))
    for i, proyeccion in enumerate(img_fan):
        for j, rayo in enumerate(proyeccion):
            paralelo[i+j, int(D*np.sin(j))] = rayo
    return paralelo


if __name__ == '__main__':
    from rayosParalelos import *

    img = data.shepp_logan_phantom()

    img_fan = trans_fan(img, 1, 1, 300)
    img_paral = fan2paralelo(img_fan, 200)
    #img_paral /= img_paral.max()
    #img_paral = filters.rank.mean(img_paral, morphology.disk(30))
    img_paral = filtro_hamming(img_paral)
    recon = inv_radon(img_paral)

    imgs = [img, img_fan, img_paral, recon]
    for i, img in enumerate(imgs):
        plt.subplot(1,len(imgs)+1,i+1)
        plt.imshow(img, cmap='gray')
    plt.tight_layout()
    plt.show()