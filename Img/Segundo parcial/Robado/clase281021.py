import numpy as np
import matplotlib.pyplot as plt
from skimage import filters,exposure, data
import skimage.filters.thresholding as thr
from skimage.morphology import binary_closing,disk,remove_small_objects

def normaliza(ima,inf=0,sup=1):
    ima = ima - np.min(ima)
    ima = ima/np.max(ima)
    ima = ima*(sup-inf) + inf
    return ima 

def Homomorfico(fil, col, fc, y_l, y_h, c):
    filtro = np.zeros([fil, col])
    for i in range(fil):
        for j in range(col):
            circulo = ((fil/2) - i)**2 + ((col/2) - j)**2
            filtro[i,j] = (y_h - y_l) * (1 - np.exp(-c * circulo / (fc**2))) + y_l
    return(filtro)

def pasabaja_but(fil,col,fcorte,N):
    filtro=np.zeros((fil,col))
    for i in range(fil):
        for j in range(col):
            circulo=np.sqrt((fil/2-i)**2+(col/2-j)**2)
            filtro[i,j]=1/(1+(circulo/fcorte)**(2*N))
    return filtro


def pasaalta_but(fil,col,fcorte,N):
    filtro=np.zeros((fil,col))
    for i in range(fil):
        for j in range(col):
            circulo=np.sqrt((fil/2-i)**2+(col/2-j)**2)
            filtro[i,j]=1/(1+(fcorte/circulo)**(2*N))
    return filtro

imagen = data.retina()/255
imagen = imagen[:,:,1]
fourier1 = np.fft.fftshift(np.fft.fft2(imagen))
filtro = pasabaja_but(imagen.shape[0], imagen.shape[1], 50,1)
convolucion = fourier1 * filtro
resultado1 =np.abs( np.fft.ifft2( np.fft.ifftshift(convolucion)))

resultado2 = normaliza(np.minimum(imagen, resultado1))
fourier2 = np.fft.fftshift(np.fft.fft2(resultado2))
filtro2 = Homomorfico(imagen.shape[0], imagen.shape[1], 10, 5, 10, 1)
convolucion2 = fourier2 * filtro2

resultado3 =np.abs( np.fft.ifft2( np.fft.ifftshift(convolucion2)))

resultado4 = normaliza(np.abs(resultado2 + resultado3))
plt.figure()
plt.imshow(imagen, cmap='gray')

plt.figure()
plt.imshow(resultado2, cmap='gray')

plt.figure()
plt.imshow(resultado3, cmap='gray')

plt.figure()
plt.imshow(resultado4, cmap='gray')
plt.show()



