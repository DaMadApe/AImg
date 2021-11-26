import numpy as np
import matplotlib.pyplot as plt
from skimage import data,transform

def trans_radon(img, step=1):
    trans = []
    for theta in np.arange(0, 180, step):
        captura = np.sum(transform.rotate(img, theta), axis=0)
        trans.append(captura)
    trans = np.array(trans)
    return trans

def inv_radon(t_radon):
    recons = np.zeros(img.shape)
    ang = 180/t_radon.shape[0]
    for i, fila in enumerate(t_radon):
        proyec = fila * np.ones((img.shape[1], 1))
        recons += transform.rotate(proyec, -i*ang)
    return recons

def filtro_hamming(t_radon):
    """
    Ventana de Hamming en frecuencia para cada fila de un arreglo
    """
    """"""
    a = 1
    omega=np.arange( -np.pi, np.pi-(2*np.pi)/(t_radon.shape[1]+1), (2*np.pi)/(t_radon.shape[1]+1) )
    rn1 = np.abs( 2/a*np.sin(a*omega/2) )
    rn2 = np.sin(a*omega/2)
    rd = (a*omega)/2
    filtro = rn1 * (rn2/rd)**2
    """"""
    # c = 0.54
    # omega = np.arange(-np.pi, np.pi, t_radon.shape[1])
    # hamming = c + (c-1)*np.cos(2*np.pi*omega/t_radon.shape[1])
    fourier = np.fft.fft(t_radon)
    filtro_frec = np.fft.fftshift(filtro)
    convolucion = fourier * filtro_frec
    return np.real(np.fft.ifft(convolucion))


if __name__ == '__main__':

    img = data.shepp_logan_phantom()

    t_radon = trans_radon(img)
    suma = inv_radon(t_radon)
    filtrado = filtro_hamming(t_radon)
    final = inv_radon(filtrado)

    plt.subplot(1,4,1)
    plt.imshow(suma, cmap='gray')
    plt.subplot(1,4,2)
    plt.imshow(t_radon, cmap='gray')
    plt.subplot(1,4,3)
    plt.imshow(filtrado, cmap='gray')
    plt.subplot(1,4,4)
    plt.imshow(final, cmap='gray')
    plt.show()