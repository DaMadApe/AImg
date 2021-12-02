import numpy as np
import matplotlib.pyplot as plt
from skimage import data,transform

"""
Archivo con funciones asociadas a proyecciones de rayos en abanico
"""
def trans_fan(imagen, pasos_alfa, pasos_beta):
    h, w = imagen.shape
    proyeccion = np.zeros((pasos_alfa, pasos_beta))

    hm = h//2
    beta_lim = np.arctan(h/(2*w))
    p = 2*beta_lim/np.pi
    pasos1 = int(round((1-p) * pasos_beta/2))
    pasos2 = int(round(p * pasos_beta))

    angulos_alfa = np.linspace(0, 180, pasos_alfa) # alfa se usa en grados
    angulos_beta1 = np.linspace(np.pi/2, beta_lim+np.pi/pasos_beta, pasos1)
    angulos_beta2 = np.linspace(-beta_lim, beta_lim, pasos2)

    for a, alfa in enumerate(angulos_alfa):
        img_rot = transform.rotate(imagen, alfa)
        # Primer y tercer sectores
        for b, beta in enumerate(angulos_beta1):
            y = np.arange(0, hm)
            x = np.linspace(0, hm/np.tan(beta)-1, hm, dtype=int)
            proyeccion[a, -b-1] = img_rot[hm+y, x].sum()
            proyeccion[a, b] = img_rot[hm-y, x].sum()
        # Segundo sector
        for b, beta in enumerate(angulos_beta2):
            x = np.arange(0, w)
            y = np.linspace(0, w*np.tan(beta)-1, w, dtype=int) + hm
            proyeccion[a, b+pasos1] = img_rot[y, x].sum()
    return proyeccion

def trans_fan2(imagen, pasos_alfa, pasos_beta, D):
    h, w = imagen.shape
    proyeccion = np.zeros((pasos_alfa, pasos_beta))

    img_pad = np.pad(imagen, D)
    h, w = img_pad.shape
    hm = h//2
    beta_lim = np.arctan(h/(2*w))

    angulos_alfa = np.linspace(0, 180, pasos_alfa) # alfa se usa en grados
    angulos_beta = np.linspace(-beta_lim, beta_lim, pasos_beta)

    for a, alfa in enumerate(angulos_alfa):
        img_rot = transform.rotate(img_pad, alfa)
        # Segundo sector
        for b, beta in enumerate(angulos_beta):
            x = np.arange(0, w)
            y = np.linspace(0, w*np.tan(beta)-1, w, dtype=int) + hm
            proyeccion[a, b] = img_rot[y, x].sum()
    return proyeccion

def fan2paralelo(img_fan, D):
    a, b = img_fan.shape
    paralelo = np.zeros((a+b, b))
    for i, proyeccion in enumerate(img_fan):
        for j, rayo in enumerate(proyeccion):
            paralelo[i+j, int(b*np.sin(j))] += rayo
    return paralelo


if __name__ == '__main__':
    from rayosParalelos import *

    img = data.shepp_logan_phantom()

    img_fan = trans_fan(img, 180, 180, 300)
    img_paral = fan2paralelo(img_fan, img.shape[0])
    recon = inv_radon(img_paral)

    imgs = [img, img_fan, img_paral, recon]
    for i, img in enumerate(imgs):
        plt.subplot(1,len(imgs)+1,i+1)
        plt.imshow(img, cmap='gray')
    plt.tight_layout()
    plt.show()