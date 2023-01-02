import numpy as np
import matplotlib.pyplot as plt
from skimage import data,transform

"""
Archivo con funciones asociadas a proyecciones de rayos paralelos
"""

def trans_radon(img, step=1):
    trans = []
    for theta in np.arange(0, 180, step):
        captura = np.sum(transform.rotate(img, theta), axis=0)
        trans.append(captura)
    trans = np.array(trans)
    return trans

def inv_radon(t_radon):
    ancho = t_radon.shape[1]
    recons = np.zeros((ancho, ancho))
    ang = 180/t_radon.shape[0]
    for i, fila in enumerate(t_radon):
        proyec = fila * np.ones((ancho, 1))
        recons += transform.rotate(proyec, -i*ang)
    return recons #/np.max(recons)

def filtro_hamming(t_radon):
    """
    Ventana de Hamming en frecuencia para cada fila de un arreglo
    """
    ancho = t_radon.shape[1]
    c = 0.54
    omega = np.linspace(-np.pi, np.pi, ancho)
    ventana_cos = c - (c-1)*np.cos(2*omega/np.pi)
    rampa = np.abs(omega)
    fourier = np.fft.fft(t_radon)
    filtro = np.fft.fftshift(ventana_cos * rampa)
    convolucion = fourier * filtro
    return np.real(np.fft.ifft(convolucion))


if __name__ == '__main__':

    img = data.shepp_logan_phantom()

    t_radon = trans_radon(img, step=1)
    suma = inv_radon(t_radon)
    filtrado = filtro_hamming(t_radon)
    final = inv_radon(filtrado)

    for i, img in enumerate([suma, t_radon, filtrado, final]):
        plt.subplot(1,4,i+1)
        plt.imshow(img, cmap='gray')
    plt.show()