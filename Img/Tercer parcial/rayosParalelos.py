import numpy as np
import matplotlib.pyplot as plt
from skimage import data,transform

def t_radon(img):
    trans = []
    for theta in range(0,180):
        captura = np.sum(transform.rotate(img, theta), axis=0)
        trans.append(captura)
    trans = np.array(trans)
    return trans

def recons(t_radon):
    suma = np.zeros(img.shape)
    for i, fila in enumerate(t_radon):
        proyec = fila * np.ones((img.shape[1], 1))
        suma += transform.rotate(proyec, -i)
    return suma

def filtro_hamming(t_radon):
    """"""
    a=1
    vector=np.arange( -np.pi, np.pi-(2*np.pi)/(t_radon.shape[1]+1), (2*np.pi)/(t_radon.shape[1]+1) )
    rn1 = np.abs( 2/a*np.sin(a*vector/2) )
    rn2 = np.sin(a*vector/2)
    rd = (a*vector)/2
    filtro = rn1 * (rn2/rd)**2
    """"""

    fourier = np.fft.fft(t_radon)
    filtro_frec = np.fft.fftshift(filtro)
    convolucion = fourier * filtro_frec
    return np.real(np.fft.ifft(convolucion))


img = data.shepp_logan_phantom()

t_radon = t_radon(img)
suma = recons(t_radon)
filtrado = filtro_hamming(t_radon)
final = recons(filtrado)

plt.subplot(1,4,1)
plt.imshow(suma, cmap='gray')
plt.subplot(1,4,2)
plt.imshow(t_radon, cmap='gray')
plt.subplot(1,4,3)
plt.imshow(filtrado, cmap='gray')
plt.subplot(1,4,4)
plt.imshow(final, cmap='gray')
plt.show()